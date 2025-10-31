import argparse
import jax, jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from thrml import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import hinton_init, IsingSamplingProgram
from models.ising_grid import make_grid_ising

plt.style.use("dark_background")
LINE_COLORS = [
    "#8dd3c7",
    "#ffd92f",
    "#fdae6b",
    "#b3de69",
    "#fb8072",
    "#80b1d3",
]


def autocorr_1d(x: np.ndarray, maxlag: int = 50) -> np.ndarray:
    x = x - x.mean()
    denom = np.dot(x, x)
    ac = []
    for k in range(1, maxlag + 1):
        ac.append(np.dot(x[:-k], x[k:]) / denom)
    return np.array(ac)


def build_free_blocks(flat_nodes, checker_blocks, blocking, seed, H, W):
    if blocking == "checkerboard":
        return checker_blocks
    if blocking == "random":
        rng = np.random.default_rng(seed + 12345)
        idx = np.arange(len(flat_nodes))
        rng.shuffle(idx)
        half = len(idx) // 2
        b1 = [flat_nodes[i] for i in idx[:half]]
        b2 = [flat_nodes[i] for i in idx[half:]]
        return [Block(b1), Block(b2)]
    if blocking == "stripes":
        stripes = [[], []]
        for idx, node in enumerate(flat_nodes):
            col = idx % W
            stripes[col % 2].append(node)
        return [Block(stripes[0]), Block(stripes[1])]
    if blocking == "supercell":
        cells = [[], []]
        for idx, node in enumerate(flat_nodes):
            row = idx // W
            col = idx % W
            cell = ((row // 2) + (col // 2)) % 2
            cells[cell].append(node)
        return [Block(cells[0]), Block(cells[1])]
    raise ValueError("Unknown blocking strategy")


def run_chain(H=28, W=28, steps=400, sps=1, J=0.35, seed=0, blocking="checkerboard"):
    model, checker_blocks, _, flat_nodes = make_grid_ising(H, W, J=J)
    free_blocks = build_free_blocks(flat_nodes, checker_blocks, blocking, seed, H, W)
    program = IsingSamplingProgram(model, free_blocks=free_blocks, clamped_blocks=[])
    key = jax.random.key(seed)
    init = hinton_init(key, model, free_blocks, batch_shape=())
    sched = SamplingSchedule(n_warmup=0, n_samples=steps, steps_per_sample=sps)
    chain = sample_states(
        key, program, sched,
        init_state_free=init, state_clamp=[],
        nodes_to_sample=[Block(flat_nodes)]
    )[0]  # (steps, H*W)
    # magnetization per step
    m = np.array(chain.mean(axis=1))
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=28)
    parser.add_argument("--W", type=int, default=28)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--maxlag", type=int, default=50)
    parser.add_argument("--sps", type=int, default=4)
    parser.add_argument("--J", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--blockings",
        nargs="+",
        default=["checkerboard", "stripes", "supercell", "random"],
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6.8, 4.2), facecolor="#1a1a1a")
    ax.set_facecolor("#141414")
    lag_range = range(1, args.maxlag + 1)

    for idx, blocking in enumerate(args.blockings):
        seed = args.seed + idx
        magnet = run_chain(
            H=args.H,
            W=args.W,
            steps=args.steps,
            sps=args.sps,
            J=args.J,
            seed=seed,
            blocking=blocking,
        )
        ac = autocorr_1d(magnet, maxlag=args.maxlag)
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        label = f"{blocking} (sps={args.sps})"
        ax.plot(lag_range, ac, label=label, color=color, linewidth=2)

    ax.set_xlabel("Lag", color="#f0f0f0")
    ax.set_ylabel("Autocorrelation (magnetization)", color="#f0f0f0")
    ax.set_title(f"Ising 28×28 mixing vs blocking (sps={args.sps})", color="#fafafa")
    ax.tick_params(colors="#d8d8d8")
    ax.grid(True, color="#2a2a2a", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = f"outputs/autocorr_blockings_sps{args.sps}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
