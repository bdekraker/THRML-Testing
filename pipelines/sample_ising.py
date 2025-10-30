import os
import numpy as np
import imageio.v2 as imageio
import jax, jax.numpy as jnp
from PIL import Image
from thrml import Block
from thrml.models.ising import hinton_init
from thrml.block_sampling import SamplingSchedule, sample_states
from models.ising_grid import make_grid_ising


def to_uint8(x_jax: jnp.ndarray) -> np.ndarray:
    """Map spins in {-1,+1} to {0,255} and return a NumPy array on host."""
    x = jax.device_get(x_jax)              # JAX array -> host
    x = ((x + 1.0) * 127.5).astype(np.uint8)
    return np.array(x)                     # ensure plain NumPy


def tile_grid(imgs_uint8: np.ndarray, H: int, W: int, rows=4, cols=4,
              out_path="outputs/samples_ising.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tile = Image.new("L", (W*cols, H*rows))
    for idx, im in enumerate(imgs_uint8):
        r, c = divmod(idx, cols)
        tile.paste(Image.fromarray(im), (c*W, r*H))
    tile.save(out_path)
    print("Saved grid:", out_path)


def main():
    print("Devices:", jax.devices())
    H = W = 28
    model, free_blocks, program, flat_nodes = make_grid_ising(H, W, J=0.35, beta=1.0)

    key = jax.random.key(0)
    init = hinton_init(key, model, free_blocks, batch_shape=())

    # Batch of 16 samples
    schedule = SamplingSchedule(n_warmup=200, n_samples=16, steps_per_sample=2)
    samples = sample_states(
        key, program, schedule,
        init_state_free=init, state_clamp=[],
        nodes_to_sample=[Block(flat_nodes)]
    )[0]  # shape: (16, H*W)

    imgs = to_uint8(samples.reshape(-1, H, W))     # NumPy uint8
    tile_grid(imgs, H, W, out_path="outputs/samples_ising.png")

    # Short GIF of a single chain mixing
    key2 = jax.random.key(42)
    schedule_gif = SamplingSchedule(n_warmup=0, n_samples=60, steps_per_sample=1)
    chain = sample_states(
        key2, program, schedule_gif,
        init_state_free=init, state_clamp=[],
        nodes_to_sample=[Block(flat_nodes)]
    )[0]  # (60, H*W)

    frames = [Image.fromarray(to_uint8(frame.reshape(H, W))) for frame in chain]
    gif_path = "outputs/mixing.gif"
    imageio.mimsave(gif_path, frames, fps=10)
    print("Saved GIF:", gif_path)


if __name__ == "__main__":
    main()
