"""
Stage 1: JEPA pretraining (self-supervised, no labels).
Predict latent of a future frame from a window of past frames.
Online encoder trained by gradient descent; target encoder is an EMA copy
(stop-gradient), preventing representation collapse.

Requires: pip install flax optax
Run: python train_jepa.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import csv
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax
from flax.training import train_state

# CONFIG

CONFIG = dict(
    dataset_root   = "synthetic_dataset",
    res            = 256,
    latent_dim     = 256,
    min_context    = 6,
    max_context    = 12,
    max_horizon    = 15,     
    batch_size     = 8,
    steps          = 5000,
    lr             = 3e-4,
    ema_momentum   = 0.996,
    log_every      = 50,
    ckpt_every     = 500,
    ckpt_dir       = "checkpoints",
    seed           = 0,
)
os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)
print("JAX devices:", jax.devices())

CONFIG["variance_weight"] = 25.0
CONFIG["variance_target"] = 1.0 / (CONFIG["latent_dim"] ** 0.5)  # ~0.0625 for latent_dim=256;
# normalized vectors live on the unit hypersphere, so natural per-dim std is ~1/sqrt(dim),

def variance_loss(embeddings, target_std=CONFIG["variance_target"]):
    """VICReg variance term: penalizes any embedding dimension whose std across
    the batch falls below target_std. Directly opposes collapse to a constant vector."""
    std = jnp.sqrt(jnp.var(embeddings, axis=0) + 1e-4)
    return jnp.mean(jax.nn.relu(target_std - std))


# DATA: load manifest, sample (context_frames, target_frame) windows

def load_manifest(root):
    rows = []
    with open(os.path.join(root, "manifest.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(r["filepath"])
    return rows

MANIFEST = load_manifest(CONFIG["dataset_root"])
print(f"Dataset: {len(MANIFEST)} videos")

from collections import OrderedDict

_cache = OrderedDict()
_CACHE_MAX_VIDEOS = 500  # ~500 * 2.5MB ~= 1.25GB

def load_video_frames(path):
    if path in _cache:
        _cache.move_to_end(path)
        return _cache[path]
    with np.load(path) as d:
        frames = d["frames"]
    _cache[path] = frames
    _cache.move_to_end(path)
    if len(_cache) > _CACHE_MAX_VIDEOS:
        _cache.popitem(last=False)
    return frames


def sample_batch(rng, batch_size):
    """Returns context [B, T_ctx_max, RES, RES, 3], ctx_len [B], target [B, RES, RES, 3]."""
    paths = rng.choice(MANIFEST, size=batch_size, replace=True)
    ctx_lens = rng.integers(CONFIG["min_context"], CONFIG["max_context"] + 1, size=batch_size)
    T = CONFIG["max_context"]
    ctx = np.zeros((batch_size, T, CONFIG["res"], CONFIG["res"], 3), dtype=np.uint8)
    tgt = np.zeros((batch_size, CONFIG["res"], CONFIG["res"], 3), dtype=np.uint8)

    for i, p in enumerate(paths):
        frames = load_video_frames(p)
        n_frames = frames.shape[0]
        c_len = min(ctx_lens[i], n_frames - 2)
        c_len = max(c_len, 2)
        max_h = min(CONFIG["max_horizon"], n_frames - c_len - 1)
        horizon = rng.integers(1, max(max_h, 2))
        start = rng.integers(0, n_frames - c_len - horizon)
        ctx[i, :c_len] = frames[start:start + c_len]
        tgt[i] = frames[start + c_len + horizon]
        ctx_lens[i] = c_len

    return ctx.astype(np.float32) / 255.0, ctx_lens, tgt.astype(np.float32) / 255.0


# small CNN encoder + transformer predictor

class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):  # x: [B, RES, RES, 3]
        for feat in (32, 64, 128, 256, 256):
            x = nn.Conv(feat, (4, 4), strides=(2, 2), padding="SAME")(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.latent_dim)(x)
        return x


class Predictor(nn.Module):
    latent_dim: int
    n_layers: int = 2
    n_heads: int = 4

    @nn.compact
    def __call__(self, ctx_latents, ctx_mask, horizon_embed):
        
        B, T, D = ctx_latents.shape
        pos = self.param("pos_embed", nn.initializers.normal(0.02), (1, T, D))
        x = ctx_latents + pos
        attn_mask = ctx_mask[:, None, None, :].astype(bool)  
        for _ in range(self.n_layers):
            y = nn.LayerNorm()(x)
            y = nn.MultiHeadDotProductAttention(num_heads=self.n_heads)(y, y, mask=attn_mask)
            x = x + y
            y = nn.LayerNorm()(x)
            y = nn.Dense(D * 4)(y)
            y = nn.gelu(y)
            y = nn.Dense(D)(y)
            x = x + y
        # pool valid context tokens, condition on target horizon, predict target latent
        mask_f = ctx_mask[..., None].astype(jnp.float32)
        pooled = jnp.sum(x * mask_f, axis=1) / jnp.clip(jnp.sum(mask_f, axis=1), 1.0)
        h = jnp.concatenate([pooled, horizon_embed], axis=-1)
        h = nn.Dense(D * 2)(h)
        h = nn.gelu(h)
        pred = nn.Dense(D)(h)
        return pred


def horizon_embedding(horizon, dim):
    """Simple sinusoidal embedding for the prediction horizon (scalar per example)."""
    freqs = jnp.exp(jnp.linspace(0, jnp.log(1000.0), dim // 2))
    h = horizon[:, None].astype(jnp.float32)
    return jnp.concatenate([jnp.sin(h * freqs), jnp.cos(h * freqs)], axis=-1)


# TRAIN STATE (online encoder + predictor; target encoder = EMA params)

class JEPAState(train_state.TrainState):
    target_params: dict = None


def create_state(rng):
    enc = Encoder(CONFIG["latent_dim"])
    pred = Predictor(CONFIG["latent_dim"])

    dummy_img = jnp.zeros((1, CONFIG["res"], CONFIG["res"], 3))
    enc_params = enc.init(rng, dummy_img)["params"]

    dummy_ctx = jnp.zeros((1, CONFIG["max_context"], CONFIG["latent_dim"]))
    dummy_mask = jnp.ones((1, CONFIG["max_context"]))
    dummy_h = jnp.zeros((1, CONFIG["latent_dim"]))
    pred_params = pred.init(rng, dummy_ctx, dummy_mask, dummy_h)["params"]

    params = {"encoder": enc_params, "predictor": pred_params}
    tx = optax.adamw(CONFIG["lr"])
    state = JEPAState.create(
        apply_fn=None, params=params, tx=tx, target_params=params,
    )
    return state, enc, pred


# TRAIN STEP

def make_train_step(enc, pred):
    def loss_fn(params, target_params, ctx, ctx_mask, tgt_img, horizon):
        B, T = ctx_mask.shape
        ctx_flat = ctx.reshape(B * T, CONFIG["res"], CONFIG["res"], 3).astype(jnp.bfloat16)
        ctx_latents = enc.apply({"params": params["encoder"]}, ctx_flat).astype(jnp.float32)
        ctx_latents = ctx_latents.reshape(B, T, CONFIG["latent_dim"])
        
        ctx_latents = ctx_latents * ctx_mask[..., None]

        h_embed = horizon_embedding(horizon, CONFIG["latent_dim"])
        pred_latent = pred.apply({"params": params["predictor"]}, ctx_latents, ctx_mask, h_embed)

        target_latent = enc.apply({"params": target_params["encoder"]}, tgt_img.astype(jnp.bfloat16))
        target_latent = jax.lax.stop_gradient(target_latent).astype(jnp.float32)

        online_target_latent = enc.apply({"params": params["encoder"]}, tgt_img.astype(jnp.bfloat16))
        online_target_latent = online_target_latent.astype(jnp.float32)

        pred_loss = jnp.mean((pred_latent - target_latent) ** 2)

        # anti-collapse: regularize the online encoder's output on target-type images
        # and the predictor's output both in raw space, matching pred_loss above.
        var_loss = variance_loss(online_target_latent) + variance_loss(pred_latent)
        loss = pred_loss + CONFIG["variance_weight"] * var_loss

        # per-dimension std across the batch, averaged over dims.
        # near 0 => embeddings are collapsing toward a constant vector.
        target_std = jnp.mean(jnp.std(target_latent, axis=0))
        online_std = jnp.mean(jnp.std(online_target_latent, axis=0))
        return loss, (target_std, pred_loss, var_loss, online_std)

    @jax.jit
    def train_step(state, ctx, ctx_mask, tgt_img, horizon):
        (loss, (target_std, pred_loss, var_loss, online_std)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(
            state.params, state.target_params, ctx, ctx_mask, tgt_img, horizon)
        state = state.apply_gradients(grads=grads)
        new_target = jax.tree_util.tree_map(
            lambda t, o: CONFIG["ema_momentum"] * t + (1 - CONFIG["ema_momentum"]) * o,
            state.target_params, state.params,
        )
        state = state.replace(target_params=new_target)
        return state, loss, target_std, pred_loss, var_loss, online_std

    return train_step


# MAIN

if __name__ == "__main__":
    rng_np = np.random.default_rng(CONFIG["seed"])
    key = random.PRNGKey(CONFIG["seed"])
    state, enc, pred = create_state(key)
    train_step = make_train_step(enc, pred)

    T = CONFIG["max_context"]
    t0 = time.time()
    for step in range(1, CONFIG["steps"] + 1):
        ctx_np, ctx_lens, tgt_np = sample_batch(rng_np, CONFIG["batch_size"])
        ctx_mask_np = (np.arange(T)[None, :] < ctx_lens[:, None]).astype(np.float32)
        horizon_np = ctx_lens.astype(np.float32)  # placeholder scalar conditioning signal

        ctx_j = jnp.asarray(ctx_np)
        mask_j = jnp.asarray(ctx_mask_np)
        tgt_j = jnp.asarray(tgt_np)
        horizon_j = jnp.asarray(horizon_np)

        state, loss, target_std, pred_loss, var_loss, online_std = train_step(
            state, ctx_j, mask_j, tgt_j, horizon_j)

        if step % CONFIG["log_every"] == 0:
            elapsed = time.time() - t0
            print(f"step {step}/{CONFIG['steps']} loss={float(loss):.4f} "
                  f"pred={float(pred_loss):.4f} var={float(var_loss):.4f} "
                  f"target_std={float(target_std):.5f} online_std={float(online_std):.5f} "
                  f"({elapsed/step:.3f}s/step)")

        if step % CONFIG["ckpt_every"] == 0:
            path = os.path.join(CONFIG["ckpt_dir"], f"step_{step}.npz")
            flat_params = jax.tree_util.tree_map(np.asarray, state.params)
            np.savez(path, **{"encoder": flat_params["encoder"],
                               "predictor": flat_params["predictor"]}, allow_pickle=True)
            print(f"Saved checkpoint: {path}")

    print("Training done.")