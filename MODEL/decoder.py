import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax
from flax.training import train_state
from PIL import Image

from train_jepa import (
    Encoder,
    Predictor,
    horizon_embedding,
    load_manifest,
    sample_batch,
)

CKPT_PATH = "checkpoints/step_50000.npz"

JEPA_CONFIG = dict(
    dataset_root = "synthetic_dataset",
    res          = 256,
    latent_dim   = 256,
    min_context  = 6,
    max_context  = 12,
    max_horizon  = 15,
)

CONFIG = dict(
    res         = JEPA_CONFIG["res"],
    latent_dim  = JEPA_CONFIG["latent_dim"],
    batch_size  = 8,
    steps       = 4000,
    lr          = 3e-4,
    log_every   = 50,
    ckpt_every  = 500,
    ckpt_dir    = "decoder_checkpoints",
    sample_dir  = "decoder_samples",
    seed        = 1,
    fg_weight   = 15.0,  
)

# DECODER
class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z): 
        x = nn.Dense(8 * 8 * 256)(z)
        x = x.reshape(z.shape[0], 8, 8, 256)
        for feat in (256, 128, 64, 32):
            x = nn.ConvTranspose(feat, (4, 4), strides=(2, 2), padding="SAME")(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        x = nn.ConvTranspose(3, (4, 4), strides=(2, 2), padding="SAME")(x)
        return nn.sigmoid(x) 


# HELPER FUNCTIONS
def load_jepa_checkpoint(path):
    with np.load(path, allow_pickle=True) as d:
        encoder_params = d["encoder"].item()
        predictor_params = d["predictor"].item()
    return encoder_params, predictor_params


def encode_and_predict(enc, pred, encoder_params, predictor_params, ctx, ctx_mask, horizon):
    B, T = ctx_mask.shape
    ctx_flat = ctx.reshape(B * T, CONFIG["res"], CONFIG["res"], 3).astype(jnp.bfloat16)
    ctx_latents = enc.apply({"params": encoder_params}, ctx_flat).astype(jnp.float32)
    ctx_latents = ctx_latents.reshape(B, T, CONFIG["latent_dim"]) * ctx_mask[..., None]
    h_embed = horizon_embedding(horizon, CONFIG["latent_dim"])
    pred_latent = pred.apply({"params": predictor_params}, ctx_latents, ctx_mask, h_embed)
    return jax.lax.stop_gradient(pred_latent)


def make_train_step(decoder):
    bg_color = jnp.array([235, 235, 240], dtype=jnp.float32) / 255.0

    def loss_fn(dec_params, pred_latent, tgt_img):
        recon = decoder.apply({"params": dec_params}, pred_latent)
        # weight up pixels that differ from background
        diff_from_bg = jnp.sum((tgt_img - bg_color) ** 2, axis=-1, keepdims=True)  # [B,H,W,1]
        is_fg = (diff_from_bg > 0.01).astype(jnp.float32)
        weight = 1.0 + CONFIG["fg_weight"] * is_fg
        return jnp.mean(weight * (recon - tgt_img) ** 2)

    @jax.jit
    def train_step(dec_state, pred_latent, tgt_img):
        loss, grads = jax.value_and_grad(loss_fn)(dec_state.params, pred_latent, tgt_img)
        dec_state = dec_state.apply_gradients(grads=grads)
        return dec_state, loss

    return train_step


def save_sample_grid(decoder, dec_params, pred_latent, tgt_img, step):
    recon = decoder.apply({"params": dec_params}, pred_latent)
    recon = np.asarray(jnp.clip(recon, 0, 1) * 255).astype(np.uint8)
    truth = np.asarray(tgt_img * 255).astype(np.uint8)

    n = min(4, recon.shape[0])
    rows = []
    for i in range(n):
        rows.append(np.concatenate([truth[i], recon[i]], axis=1))  # side by side
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(os.path.join(CONFIG["sample_dir"], f"step_{step}.png"))

# MAIN EXECUTION
if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    
    # Setup directories
    os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)
    os.makedirs(CONFIG["sample_dir"], exist_ok=True)

    manifest = load_manifest(JEPA_CONFIG["dataset_root"])
    print(f"Dataset: {len(manifest)} videos")

    encoder_params, predictor_params = load_jepa_checkpoint(CKPT_PATH)
    enc = Encoder(CONFIG["latent_dim"])
    pred = Predictor(CONFIG["latent_dim"])

    key = random.PRNGKey(CONFIG["seed"])
    decoder = Decoder()
    dummy_z = jnp.zeros((1, CONFIG["latent_dim"]))
    dec_params = decoder.init(key, dummy_z)["params"]
    tx = optax.adamw(CONFIG["lr"])
    dec_state = train_state.TrainState.create(apply_fn=None, params=dec_params, tx=tx)

    train_step = make_train_step(decoder)
    T = JEPA_CONFIG["max_context"]
    rng_np = np.random.default_rng(CONFIG["seed"])

    t0 = time.time()
    for step in range(1, CONFIG["steps"] + 1):
        ctx_np, ctx_lens, tgt_np = sample_batch(rng_np, CONFIG["batch_size"])
        
        ctx_mask_np = (np.arange(T)[None, :] < ctx_lens[:, None]).astype(np.float32)
        horizon_np = ctx_lens.astype(np.float32) 

        ctx_j = jnp.asarray(ctx_np)
        mask_j = jnp.asarray(ctx_mask_np)
        tgt_j = jnp.asarray(tgt_np)
        horizon_j = jnp.asarray(horizon_np)

        pred_latent = encode_and_predict(
            enc, pred, encoder_params, predictor_params, ctx_j, mask_j, horizon_j
        )
        dec_state, loss = train_step(dec_state, pred_latent, tgt_j)

        if step % CONFIG["log_every"] == 0:
            elapsed = time.time() - t0
            print(f"step {step}/{CONFIG['steps']} recon_mse={float(loss):.5f} "
                  f"({elapsed/step:.3f}s/step)")

        if step % CONFIG["ckpt_every"] == 0:
            path = os.path.join(CONFIG["ckpt_dir"], f"decoder_step_{step}.npz")
            flat_params = jax.tree_util.tree_map(np.asarray, dec_state.params)
            np.savez(path, decoder=flat_params, allow_pickle=True)
            save_sample_grid(decoder, dec_state.params, pred_latent, tgt_j, step)
            print(f"Saved checkpoint + sample grid at step {step}")

    print("Decoder training done. Check", CONFIG["sample_dir"], "for reconstructions.")
    
    
"""
Stage 3: decoder training (encoder + predictor frozen).
Loads a Stage 1 JEPA checkpoint, freezes it, trains only a decoder to
reconstruct the target frame's pixels from the predicted latent.
"""