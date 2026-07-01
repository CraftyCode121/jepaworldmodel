import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false") #Don't reserve all the GPU memory up front

import json
import time
import csv
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
from multiprocessing import Pool
from config import CONFIG

print("JAX devices:", jax.devices())
os.makedirs(CONFIG["dataset_root"], exist_ok=True)

# PER-FAMILY VIDEO COUNTS (weighted, sums exactly to total_videos)

def allocate_counts(weights, total):
    names = list(weights.keys())
    w = np.array([weights[n] for n in names], dtype=np.float64)
    raw = w / w.sum() * total
    counts = np.floor(raw).astype(int)
    remainder = total - counts.sum()
    # give leftover videos to families with the largest fractional remainder
    frac_order = np.argsort(-(raw - counts))
    for i in range(remainder):
        counts[frac_order[i % len(names)]] += 1
    return {n: int(c) for n, c in zip(names, counts)}

VIDEO_COUNTS = allocate_counts(CONFIG["family_weights"], CONFIG["total_videos"])
print("Per-family video counts:", VIDEO_COUNTS)

RES = CONFIG["res"]
N_FRAMES = CONFIG["n_frames"]
DT = CONFIG["dt"]
MAX_OBJ = 2  # every family standardized to <=2 objects; unused slots marked inactive


# FAMILY PHYSICS : each returns pos[F,MAX_OBJ,2], vel[F,MAX_OBJ,2],
# mass[MAX_OBJ], radius[MAX_OBJ], active[MAX_OBJ] (1=active, 0=unused slot)

FLOOR, CEIL, LEFT, RIGHT = 245.0, 10.0, 10.0, 245.0
G = 200.0

def _pad(obj0_pos, obj0_vel, obj0_mass, obj0_radius, obj1=None):
    if obj1 is None:
        pos = jnp.stack([obj0_pos, jnp.zeros_like(obj0_pos)], axis=1)
        vel = jnp.stack([obj0_vel, jnp.zeros_like(obj0_vel)], axis=1)
        mass = jnp.stack([obj0_mass, jnp.array(0.0)])
        radius = jnp.stack([obj0_radius, jnp.array(0.0)])
        active = jnp.array([1.0, 0.0])
    else:
        pos = jnp.stack([obj0_pos, obj1[0]], axis=1)
        vel = jnp.stack([obj0_vel, obj1[1]], axis=1)
        mass = jnp.stack([obj0_mass, obj1[2]])
        radius = jnp.stack([obj0_radius, obj1[3]])
        active = jnp.array([1.0, 1.0])
    return pos, vel, mass, radius, active


def fam_projectile(key, n, dt):
    k = random.split(key, 5)
    x0 = random.uniform(k[0], (), minval=20, maxval=60)
    y0 = random.uniform(k[1], (), minval=100, maxval=180)
    vx0 = random.uniform(k[2], (), minval=40, maxval=90)
    vy0 = random.uniform(k[3], (), minval=-20, maxval=20)
    mass = random.uniform(k[4], (), minval=0.5, maxval=3.0)
    restitution = 0.7

    def step(c, _):
        y, vy = c
        vy = vy + G * dt
        y_new = y + vy * dt
        b = y_new > FLOOR
        y_new = jnp.where(b, FLOOR, y_new)
        vy_new = jnp.where(b, -vy * restitution, vy)
        return (y_new, vy_new), (y_new, vy_new)

    (_, _), (y_seq, vy_seq) = lax.scan(step, (y0, vy0), None, length=n)
    t = jnp.arange(n, dtype=jnp.float32) * dt
    x_seq = x0 + vx0 * t
    vx_seq = jnp.full((n,), vx0)
    pos = jnp.stack([x_seq, y_seq], axis=1)
    vel = jnp.stack([vx_seq, vy_seq], axis=1)
    return _pad(pos, vel, mass, jnp.array(10.0))


def fam_pendulum(key, n, dt):
    k = random.split(key, 4)
    cx = random.uniform(k[0], (), minval=100, maxval=156)
    cy = 40.0
    length = random.uniform(k[1], (), minval=80, maxval=150)
    theta0 = random.uniform(k[2], (), minval=-1.2, maxval=1.2)
    mass = random.uniform(k[3], (), minval=0.5, maxval=3.0)
    g_l = 900.0 / length

    def step(c, _):
        th, om = c
        al = -g_l * jnp.sin(th)
        om = om + al * dt
        th = th + om * dt
        return (th, om), (th, om)

    (_, _), (th_seq, om_seq) = lax.scan(step, (theta0, 0.0), None, length=n)
    x_seq = cx + length * jnp.sin(th_seq)
    y_seq = cy + length * jnp.cos(th_seq)
    vx_seq = length * om_seq * jnp.cos(th_seq)
    vy_seq = -length * om_seq * jnp.sin(th_seq)
    pos = jnp.stack([x_seq, y_seq], axis=1)
    vel = jnp.stack([vx_seq, vy_seq], axis=1)
    return _pad(pos, vel, mass, jnp.array(10.0))


def fam_free_fall(key, n, dt):
    k = random.split(key, 3)
    x0 = random.uniform(k[0], (), minval=40, maxval=216)
    y0 = random.uniform(k[1], (), minval=30, maxval=100)
    mass = random.uniform(k[2], (), minval=0.5, maxval=3.0)
    restitution = 0.6

    def step(c, _):
        y, vy = c
        vy = vy + G * dt
        y_new = y + vy * dt
        b = y_new > FLOOR
        y_new = jnp.where(b, FLOOR, y_new)
        vy_new = jnp.where(b, -vy * restitution, vy)
        return (y_new, vy_new), (y_new, vy_new)

    (_, _), (y_seq, vy_seq) = lax.scan(step, (y0, 0.0), None, length=n)
    x_seq = jnp.full((n,), x0)
    vx_seq = jnp.zeros((n,))
    pos = jnp.stack([x_seq, y_seq], axis=1)
    vel = jnp.stack([vx_seq, vy_seq], axis=1)
    return _pad(pos, vel, mass, jnp.array(10.0))


def fam_circular_orbit(key, n, dt):
    k = random.split(key, 4)
    cx = random.uniform(k[0], (), minval=100, maxval=156)
    cy = random.uniform(k[1], (), minval=100, maxval=156)
    r = random.uniform(k[2], (), minval=40, maxval=90)
    omega = random.uniform(k[3], (), minval=1.0, maxval=3.0)
    mass = jnp.array(1.0)
    t = jnp.arange(n, dtype=jnp.float32) * dt
    x_seq = cx + r * jnp.cos(omega * t)
    y_seq = cy + r * jnp.sin(omega * t)
    vx_seq = -r * omega * jnp.sin(omega * t)
    vy_seq = r * omega * jnp.cos(omega * t)
    pos = jnp.stack([x_seq, y_seq], axis=1)
    vel = jnp.stack([vx_seq, vy_seq], axis=1)
    return _pad(pos, vel, mass, jnp.array(9.0))


def fam_spring_mass(key, n, dt):
    k = random.split(key, 4)
    cx = random.uniform(k[0], (), minval=90, maxval=166)
    amp = random.uniform(k[1], (), minval=30, maxval=70)
    omega = random.uniform(k[2], (), minval=2.0, maxval=5.0)
    mass = random.uniform(k[3], (), minval=0.5, maxval=3.0)
    y0 = 128.0
    t = jnp.arange(n, dtype=jnp.float32) * dt
    x_seq = cx + amp * jnp.cos(omega * t)
    y_seq = jnp.full((n,), y0)
    vx_seq = -amp * omega * jnp.sin(omega * t)
    vy_seq = jnp.zeros((n,))
    pos = jnp.stack([x_seq, y_seq], axis=1)
    vel = jnp.stack([vx_seq, vy_seq], axis=1)
    return _pad(pos, vel, mass, jnp.array(10.0))


def fam_damped_oscillation(key, n, dt):
    k = random.split(key, 4)
    cx = random.uniform(k[0], (), minval=90, maxval=166)
    amp = random.uniform(k[1], (), minval=40, maxval=80)
    omega = random.uniform(k[2], (), minval=3.0, maxval=6.0)
    damping = random.uniform(k[3], (), minval=0.5, maxval=1.5)
    mass = jnp.array(1.0)
    y0 = 128.0
    t = jnp.arange(n, dtype=jnp.float32) * dt
    env = jnp.exp(-damping * t)
    x_seq = cx + amp * env * jnp.cos(omega * t)
    y_seq = jnp.full((n,), y0)
    vx_seq = amp * env * (-damping * jnp.cos(omega * t) - omega * jnp.sin(omega * t))
    vy_seq = jnp.zeros((n,))
    pos = jnp.stack([x_seq, y_seq], axis=1)
    vel = jnp.stack([vx_seq, vy_seq], axis=1)
    return _pad(pos, vel, mass, jnp.array(10.0))


def fam_friction_slide(key, n, dt):
    k = random.split(key, 4)
    x0 = random.uniform(k[0], (), minval=20, maxval=60)
    v0 = random.uniform(k[1], (), minval=60, maxval=140)
    mu = random.uniform(k[2], (), minval=0.3, maxval=1.2)
    mass = random.uniform(k[3], (), minval=0.5, maxval=3.0)
    y0 = FLOOR
    decel = mu * G

    def step(c, _):
        x, v = c
        v_new = jnp.where(v > 0, jnp.maximum(v - decel * dt, 0.0), 0.0)
        x_new = x + v_new * dt
        return (x_new, v_new), (x_new, v_new)

    (_, _), (x_seq, v_seq) = lax.scan(step, (x0, v0), None, length=n)
    y_seq = jnp.full((n,), y0)
    vy_seq = jnp.zeros((n,))
    pos = jnp.stack([x_seq, y_seq], axis=1)
    vel = jnp.stack([v_seq, vy_seq], axis=1)
    return _pad(pos, vel, mass, jnp.array(10.0))


def fam_inclined_slide(key, n, dt):
    k = random.split(key, 4)
    angle = random.uniform(k[0], (), minval=0.3, maxval=0.9)  
    s0 = random.uniform(k[1], (), minval=0.0, maxval=20.0)    
    mu = random.uniform(k[2], (), minval=0.05, maxval=0.35)
    mass = random.uniform(k[3], (), minval=0.5, maxval=3.0)
    accel = G * (jnp.sin(angle) - mu * jnp.cos(angle))
    accel = jnp.maximum(accel, 5.0) 
    top_x, top_y = 40.0, 30.0

    def step(c, _):
        s, v = c
        v_new = v + accel * dt
        s_new = s + v_new * dt
        return (s_new, v_new), (s_new, v_new)

    (_, _), (s_seq, v_seq) = lax.scan(step, (s0, 0.0), None, length=n)
    s_max = 260.0
    s_seq = jnp.clip(s_seq, 0.0, s_max)
    x_seq = top_x + s_seq * jnp.cos(angle)
    y_seq = top_y + s_seq * jnp.sin(angle)
    vx_seq = v_seq * jnp.cos(angle)
    vy_seq = v_seq * jnp.sin(angle)
    pos = jnp.stack([x_seq, y_seq], axis=1)
    vel = jnp.stack([vx_seq, vy_seq], axis=1)
    return _pad(pos, vel, mass, jnp.array(10.0))


def fam_elastic_wall_bounce(key, n, dt):
    k = random.split(key, 5)
    x0 = random.uniform(k[0], (), minval=60, maxval=196)
    y0 = random.uniform(k[1], (), minval=60, maxval=196)
    vx0 = random.uniform(k[2], (), minval=-120, maxval=120)
    vy0 = random.uniform(k[3], (), minval=-120, maxval=120)
    mass = random.uniform(k[4], (), minval=0.5, maxval=3.0)
    radius = 10.0

    def step(c, _):
        x, y, vx, vy = c
        x_new = x + vx * dt
        y_new = y + vy * dt
        hit_x = (x_new < LEFT + radius) | (x_new > RIGHT - radius)
        hit_y = (y_new < CEIL + radius) | (y_new > FLOOR - radius)
        vx_new = jnp.where(hit_x, -vx, vx)
        vy_new = jnp.where(hit_y, -vy, vy)
        x_new = jnp.clip(x_new, LEFT + radius, RIGHT - radius)
        y_new = jnp.clip(y_new, CEIL + radius, FLOOR - radius)
        return (x_new, y_new, vx_new, vy_new), (x_new, y_new, vx_new, vy_new)

    (_, _, _, _), (x_seq, y_seq, vx_seq, vy_seq) = lax.scan(
        step, (x0, y0, vx0, vy0), None, length=n)
    pos = jnp.stack([x_seq, y_seq], axis=1)
    vel = jnp.stack([vx_seq, vy_seq], axis=1)
    return _pad(pos, vel, mass, jnp.array(radius))


def fam_two_body_collision(key, n, dt):
    k = random.split(key, 6)
    x1 = random.uniform(k[0], (), minval=30, maxval=80)
    x2 = random.uniform(k[1], (), minval=176, maxval=226)
    y0 = 128.0
    v1 = random.uniform(k[2], (), minval=30, maxval=80)
    v2 = -random.uniform(k[3], (), minval=30, maxval=80)
    m1 = random.uniform(k[4], (), minval=0.5, maxval=3.0)
    m2 = random.uniform(k[5], (), minval=0.5, maxval=3.0)
    r1 = r2 = 10.0

    def step(c, _):
        x1, x2, v1, v2 = c
        x1n = x1 + v1 * dt
        x2n = x2 + v2 * dt
        dist = x2n - x1n
        colliding = (dist <= (r1 + r2)) & (v1 > v2)
        # 1D elastic collision
        v1c = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
        v2c = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
        v1n = jnp.where(colliding, v1c, v1)
        v2n = jnp.where(colliding, v2c, v2)
        return (x1n, x2n, v1n, v2n), (x1n, x2n, v1n, v2n)

    (_, _, _, _), (x1_seq, x2_seq, v1_seq, v2_seq) = lax.scan(
        step, (x1, x2, v1, v2), None, length=n)
    y_seq = jnp.full((n,), y0)
    zero = jnp.zeros((n,))
    pos1 = jnp.stack([x1_seq, y_seq], axis=1)
    vel1 = jnp.stack([v1_seq, zero], axis=1)
    pos2 = jnp.stack([x2_seq, y_seq], axis=1)
    vel2 = jnp.stack([v2_seq, zero], axis=1)
    return _pad(pos1, vel1, m1, jnp.array(r1), obj1=(pos2, vel2, m2, jnp.array(r2)))


FAMILIES = {
    "projectile": fam_projectile,
    "pendulum": fam_pendulum,
    "free_fall": fam_free_fall,
    "circular_orbit": fam_circular_orbit,
    "spring_mass": fam_spring_mass,
    "damped_oscillation": fam_damped_oscillation,
    "friction_slide": fam_friction_slide,
    "inclined_slide": fam_inclined_slide,
    "elastic_wall_bounce": fam_elastic_wall_bounce,
    "two_body_collision": fam_two_body_collision,
}

# RENDER : generic for up to MAX_OBJ objects, jit+vmap batched

OBJ_COLORS = jnp.array([[200, 60, 60], [60, 90, 200]], dtype=jnp.float32) / 255.0
BG = jnp.array([235, 235, 240], dtype=jnp.float32) / 255.0

@jit
def _render_single(pos, radius, active):
    """pos: [F,MAX_OBJ,2] radius/active: [MAX_OBJ] -> frames [F,RES,RES,3]"""
    yy, xx = jnp.meshgrid(jnp.arange(RES, dtype=jnp.float32),
                           jnp.arange(RES, dtype=jnp.float32), indexing='ij')
    frame = jnp.broadcast_to(BG, (pos.shape[0], RES, RES, 3))

    def draw_obj(frame, obj_idx):
        cx = pos[:, obj_idx, 0][:, None, None]
        cy = pos[:, obj_idx, 1][:, None, None]
        r = radius[obj_idx]
        a = active[obj_idx]
        dist = jnp.sqrt((xx[None] - cx) ** 2 + (yy[None] - cy) ** 2)
        inside = (dist <= r) & (a > 0)
        shade = jnp.clip(1.0 - dist / (r + 1e-6), 0.0, 1.0)
        brightness = jnp.where(inside, 0.6 + 0.4 * shade, 0.0)[..., None]
        mask = inside.astype(jnp.float32)[..., None]
        color = OBJ_COLORS[obj_idx]
        return frame * (1 - mask) + color * brightness

    for i in range(MAX_OBJ):
        frame = draw_obj(frame, i)
    return frame


render_batch_jax = jit(vmap(_render_single))


def render_in_chunks(pos, radius, active, chunk_size):
    n = pos.shape[0]
    out = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = render_batch_jax(pos[start:end], radius[start:end], active[start:end])
        out.append(np.asarray((chunk * 255).astype(jnp.uint8)))
    return np.concatenate(out, axis=0)


# WRITE WORKER

def _write_one(args):
    frames_i, pos_i, vel_i, mass_i, radius_i, active_i, family, path = args
    np.savez_compressed(
        path, frames=frames_i, pos=pos_i, vel=vel_i, mass=mass_i,
        radius=radius_i, active=active_i, family=family,
    )


# RUN ONE FAMILY

def build_batch_sim(single_fn):
    @jit
    def batch_fn(keys):
        return vmap(lambda k: single_fn(k, N_FRAMES, DT))(keys)
    return batch_fn


def run_family(name, sim_fn, n_videos, seed, family_dir, manifest_rows):
    t0 = time.time()
    keys = random.split(random.PRNGKey(seed), n_videos)
    pos, vel, mass, radius, active = sim_fn(keys)
    jax.block_until_ready(pos)
    t1 = time.time()

    frames_np = render_in_chunks(pos, radius, active, CONFIG["render_chunk"])
    t2 = time.time()

    pos_np, vel_np = np.asarray(pos), np.asarray(vel)
    mass_np, radius_np, active_np = np.asarray(mass), np.asarray(radius), np.asarray(active)

    jobs = []
    for i in range(n_videos):
        fname = f"{name}_{i:06d}.npz"
        path = os.path.join(family_dir, fname)
        jobs.append((frames_np[i], pos_np[i], vel_np[i], mass_np[i],
                     radius_np[i], active_np[i], name, path))
        manifest_rows.append((f"{name}_{i:06d}", name, path, N_FRAMES))

    with Pool(CONFIG["n_write_workers"]) as pool:
        pool.map(_write_one, jobs)
    t3 = time.time()

    print(f"[{name}] n={n_videos} sim={t1-t0:.2f}s render={t2-t1:.2f}s "
          f"write={t3-t2:.2f}s total={t3-t0:.2f}s ({(t3-t0)/n_videos*1000:.1f} ms/video)")


# MAIN

if __name__ == "__main__":
    with open(os.path.join(CONFIG["dataset_root"], "config.json"), "w") as f:
        json.dump({**CONFIG, "video_counts": VIDEO_COUNTS}, f, indent=2)

    manifest_rows = []
    overall_t0 = time.time()

    for i, (name, sim_fn) in enumerate(FAMILIES.items()):
        family_dir = os.path.join(CONFIG["dataset_root"], name)
        os.makedirs(family_dir, exist_ok=True)
        batch_sim = build_batch_sim(sim_fn)
        n_videos = VIDEO_COUNTS[name]
        run_family(name, batch_sim, n_videos, seed=CONFIG["seed"] + i,
                   family_dir=family_dir, manifest_rows=manifest_rows)

    with open(os.path.join(CONFIG["dataset_root"], "manifest.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "family", "filepath", "n_frames"])
        writer.writerows(manifest_rows)

    print(f"\nDone. {len(manifest_rows)} videos generated in "
          f"{time.time() - overall_t0:.1f}s. See {CONFIG['dataset_root']}/")
    
"""
Synthetic physics dataset generator for JEPA world-model training.
10 motion families, weighted video counts (harder families get more), raw
pixel + state storage (no video codec — we train on tensors, not mp4s).

Folder layout:
  DATASET_ROOT/
    config.json                        <> full run config + per-family counts
    manifest.csv                       <> video_id, family, filepath, n_frames
    <family_name>/
      <family_name>_000000.npz         <> frames [F,RES,RES,3] uint8, pos, vel,
                                           mass, radius, active mask, family label
      

Run: python generate_data.py
"""