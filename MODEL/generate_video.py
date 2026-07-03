import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
import jax.numpy as jnp
import imageio

from train_jepa import Encoder, Predictor, horizon_embedding, load_manifest
from decoder import Decoder

# Checkpoint Paths
JEPA_CKPT    = "checkpoints/step_50000.npz"
DECODER_CKPT = "decoder_checkpoints/decoder_step_4000.npz"

#JEPA CONFIG
JEPA_CONFIG = dict(
    dataset_root = "synthetic_dataset",
    res          = 256,
    latent_dim   = 256,
    min_context  = 6,
    max_context  = 12,
    max_horizon  = 15,
)

# Generation Settings 
N_VIDEOS     = 5            
CONTEXT_LEN  = 10          
ROLLOUT_LEN  = 20          
HORIZON_STEP = 1            
OUT_DIR      = "generated_videos"
FPS          = 15

# Inherit architectural constants directly from Stage 1/2 configurations
RES          = JEPA_CONFIG["res"]
LATENT_DIM   = JEPA_CONFIG["latent_dim"]
MAX_CONTEXT  = JEPA_CONFIG["max_context"]
DATASET_ROOT = JEPA_CONFIG["dataset_root"]


# AUTOREGRESSIVE ROLLOUT LOGIC

def rollout_one(frames, encode_fn, predict_fn, decode_fn):
    
    ctx_imgs = frames[:CONTEXT_LEN].astype(np.float32) / 255.0  
    generated = [(ctx_imgs[i] * 255).astype(np.uint8) for i in range(CONTEXT_LEN)]

    latents = encode_fn(jnp.asarray(ctx_imgs)) 
    latents = list(latents)

    for step in range(ROLLOUT_LEN):
        window = latents[-MAX_CONTEXT:]
        T_valid = len(window)
        
        ctx = jnp.zeros((1, MAX_CONTEXT, LATENT_DIM))
        ctx = ctx.at[0, :T_valid].set(jnp.stack(window))
        
        mask = jnp.zeros((1, MAX_CONTEXT))
        mask = mask.at[0, :T_valid].set(1.0)
        
        h_embed = horizon_embedding(jnp.array([float(HORIZON_STEP)]), LATENT_DIM)

        next_latent = predict_fn(ctx, mask, h_embed)[0]  # [D]
        latents.append(next_latent)

        frame = decode_fn(next_latent[None])[0]  
        generated.append((np.asarray(jnp.clip(frame, 0, 1)) * 255).astype(np.uint8))

    return generated


# MAIN RUNTIME EXECUTION
if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading Stage 1 JEPA weights from: {JEPA_CKPT}")
    with np.load(JEPA_CKPT, allow_pickle=True) as d:
        encoder_params = d["encoder"].item()
        predictor_params = d["predictor"].item()
        
    print(f"Loading Stage 3 Decoder weights from: {DECODER_CKPT}")
    with np.load(DECODER_CKPT, allow_pickle=True) as d:
        decoder_params = d["decoder"].item()

    # Instantiate networks
    enc = Encoder(LATENT_DIM)
    pred = Predictor(LATENT_DIM)
    decoder = Decoder()

    # Compile JAX execution functions
    print("Compiling JAX pipelines...")
    encode = jax.jit(lambda imgs: enc.apply({"params": encoder_params}, imgs.astype(jnp.bfloat16)).astype(jnp.float32))
    predict = jax.jit(lambda ctx, mask, h: pred.apply({"params": predictor_params}, ctx, mask, h))
    decode = jax.jit(lambda z: decoder.apply({"params": decoder_params}, z))

    # Fetch validation paths from dataset manifest
    manifest = load_manifest(DATASET_ROOT)
    rng = np.random.default_rng(123)
    seed_paths = rng.choice(manifest, size=N_VIDEOS, replace=False)

    # Execute generation loop
    print(f"Starting rollout generation for {N_VIDEOS} target samples...")
    for i, path in enumerate(seed_paths):
        with np.load(path) as d:
            frames = d["frames"]
            family = str(d["family"])
            
        generated_frames = rollout_one(frames, encode, predict, decode)
        
        out_path = os.path.join(OUT_DIR, f"rollout_{i:02d}_{family}.mp4")
        imageio.mimsave(out_path, generated_frames, fps=FPS, quality=6)
        print(f"Saved {out_path} ({len(generated_frames)} total frames: {CONTEXT_LEN} seed context + {ROLLOUT_LEN} predicted)")

    print(f"Done! Check the video outputs inside: {OUT_DIR}/")
    
"""
Generate videos from the trained world model: seed with a real context window,
then autoregressively predict future latents and decode each to pixels.
"""