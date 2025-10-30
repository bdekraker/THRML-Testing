import os, numpy as np
from PIL import Image, ImageDraw
import jax, jax.numpy as jnp
from thrml import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import hinton_init, IsingSamplingProgram
from models.ising_grid import make_grid_ising
from sklearn.datasets import fetch_openml


def load_one_mnist_binarized(thresh=0.5, seed=7):
    X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    rng = np.random.RandomState(seed)
    x = X[rng.randint(0, X.shape[0])].reshape(28, 28).astype(np.float32) / 255.0
    x = (x > thresh).astype(np.float32)
    return x * 2.0 - 1.0  # {-1,+1}


def make_mask(H=28, W=28, kind="center_box"):
    m = np.zeros((H,W), dtype=np.bool_)
    if kind == "center_box":
        h, w = H//2, W//2
        r0, c0 = (H-h)//2, (W-w)//2
        m[r0:r0+h, c0:c0+w] = True
    elif kind == "grid_holes":
        m[::2, ::2] = True
    else:
        raise ValueError("unknown mask kind")
    return m  # True => keep/clamp


def spin_to_u8(x): return ((x + 1.0) * 127.5).astype(np.uint8)


def label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle([0,0,img.width,10], fill=0)  # header bar
    draw.text((2,0), text, fill=255)
    return img


def make_collage(orig_u8, masked_u8, comps_u8, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    H, W = orig_u8.shape
    cols, rows = 5, 2  # [Original, Masked] + 8 completions
    tile = Image.new("L", (W*cols, H*rows), color=255)

    # Row 0: original + masked
    tile.paste(label(Image.fromarray(orig_u8),   "original"), (0*W, 0*H))
    tile.paste(label(Image.fromarray(masked_u8), "masked"),   (1*W, 0*H))

    # Row 0 completions 1–3
    for i in range(3):
        tile.paste(label(Image.fromarray(comps_u8[i]), f"comp{i+1}"), ((i+2)*W, 0*H))

    # Row 1 completions 4–8
    for i in range(5):
        tile.paste(label(Image.fromarray(comps_u8[i+3]), f"comp{i+4}"), (i*W, 1*H))

    tile.save(out_path)
    print("Saved:", out_path)


def main():
    print("Devices:", jax.devices())
    H = W = 28; J = 0.35

    # data & mask
    x = load_one_mnist_binarized(thresh=0.5, seed=7)
    mask = make_mask(H, W, kind="center_box")

    # model & disjoint blocks (free excludes clamped)
    model, _, _, flat = make_grid_ising(H, W, J=J)
    evens, odds, clamped = [], [], []
    for i in range(H):
        for j in range(W):
            node = flat[i*W + j]
            if mask[i,j]: clamped.append(node)
            else: (evens if (i+j)%2==0 else odds).append(node)
    free_blocks = [Block(evens), Block(odds)]
    clamped_block = Block(clamped)
    program = IsingSamplingProgram(model, free_blocks=free_blocks, clamped_blocks=[clamped_block])

    # clamp values must be bool for SpinNode
    clamp_vals = (x[mask] > 0).astype(np.bool_)
    state_clamp = [jnp.array(clamp_vals)]

    key = jax.random.key(0)
    init = hinton_init(key, model, free_blocks, batch_shape=())
    schedule = SamplingSchedule(n_warmup=350, n_samples=8, steps_per_sample=2)
    samples = sample_states(key, program, schedule, init_state_free=init,
                            state_clamp=state_clamp, nodes_to_sample=[Block(flat)])[0]
    comps = np.array(samples.reshape(-1, H, W))

    orig_u8  = spin_to_u8(x)
    masked_u8 = spin_to_u8(np.where(mask, x, -1.0))
    comps_u8 = spin_to_u8(comps)

    make_collage(orig_u8, masked_u8, comps_u8, "outputs/inpaint_collage.png")


if __name__ == "__main__":
    main()
