import jax, jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from thrml import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import hinton_init
from models.ising_grid import make_grid_ising

plt.style.use("dark_background")
LINE_COLORS = ("#8dd3c7", "#ffd92f")


def autocorr_1d(x: np.ndarray, maxlag: int = 50) -> np.ndarray:
    x = x - x.mean()
    denom = np.dot(x, x)
    ac = []
    for k in range(1, maxlag + 1):
        ac.append(np.dot(x[:-k], x[k:]) / denom)
    return np.array(ac)


def run_chain(H=28, W=28, steps=400, sps=1, J=0.35, seed=0):
    model, free_blocks, program, flat_nodes = make_grid_ising(H, W, J=J)
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
    m1 = run_chain(steps=400, sps=1, J=0.35, seed=0)
    m2 = run_chain(steps=400, sps=2, J=0.35, seed=1)
    ac1 = autocorr_1d(m1, maxlag=50)
    ac2 = autocorr_1d(m2, maxlag=50)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor="#1a1a1a")
    ax.set_facecolor("#141414")
    lag_range = range(1, 51)
    ax.plot(lag_range, ac1, label="steps_per_sample=1", color=LINE_COLORS[0], linewidth=2)
    ax.plot(lag_range, ac2, label="steps_per_sample=2", color=LINE_COLORS[1], linewidth=2)
    ax.set_xlabel("Lag", color="#f0f0f0")
    ax.set_ylabel("Autocorrelation (magnetization)", color="#f0f0f0")
    ax.set_title("Ising 28×28 mixing (CPU, THRML)", color="#fafafa")
    ax.tick_params(colors="#d8d8d8")
    ax.grid(True, color="#2a2a2a", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("outputs/autocorr.png", dpi=150)
    print("Saved outputs/autocorr.png")


if __name__ == "__main__":
    main()
