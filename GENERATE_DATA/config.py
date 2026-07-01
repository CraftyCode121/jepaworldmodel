from multiprocessing import cpu_count

# CONFIG : everything tunable lives here

CONFIG = dict(
    dataset_root = "synthetic_dataset",
    total_videos = 10_000,
    fps          = 15,
    duration_s   = 3,
    res          = 256,
    seed         = 42,
    render_chunk = 10,      # videos per GPU render call
    n_write_workers = max(cpu_count() - 1, 1),

    # relative difficulty weight per family => more weight = more videos.
    # "harder" = more complex dynamics (collisions, multi-wall bounces, friction regimes)
    
    family_weights = {
        "projectile":         1.0,
        "pendulum":           1.0,
        "free_fall":          1.0,
        "circular_orbit":     1.0,
        "spring_mass":        1.0,
        "damped_oscillation": 1.0,
        "friction_slide":     1.0,
        "inclined_slide":     1.5,
        "elastic_wall_bounce":1.5,
        "two_body_collision": 1.5,
    },
)

CONFIG["n_frames"] = CONFIG["fps"] * CONFIG["duration_s"]
CONFIG["dt"] = 1.0 / CONFIG["fps"]