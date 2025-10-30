import os, time, numpy as np, jax
from thrml import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import IsingSamplingProgram, hinton_init
from models.ising_grid import make_grid_ising


def build_random_blocks(flat_nodes, rng: np.random.RandomState):
    idx = np.arange(len(flat_nodes))
    rng.shuffle(idx)
    half = len(idx)//2
    b1 = [flat_nodes[i] for i in idx[:half]]
    b2 = [flat_nodes[i] for i in idx[half:]]
    return [Block(b1), Block(b2)]


def eff_samples_per_sec(model, program, flat_nodes, free_blocks,
                        steps=128, warmup=100, sps=1, seed=0):
    key = jax.random.key(seed)
    init = hinton_init(key, model, free_blocks, batch_shape=())
    sched = SamplingSchedule(n_warmup=warmup, n_samples=steps, steps_per_sample=sps)
    t0 = time.time()
    samples = sample_states(key, program, sched,
                            init_state_free=init, state_clamp=[],
                            nodes_to_sample=[Block(flat_nodes)])[0]
    dt = time.time() - t0
    thr = steps / dt
    m = np.array(samples.mean(axis=1))  # magnetization time-series
    return thr, m


def lag1_autocorr(x: np.ndarray) -> float:
    x = x - x.mean()
    return float(np.dot(x[:-1], x[1:]) / np.dot(x, x))


def main():
    os.makedirs("outputs", exist_ok=True)

    H = W = 28
    # Checkerboard program
    model, free_blocks_chk, _, flat_nodes = make_grid_ising(H, W, J=0.35)
    prog_chk = IsingSamplingProgram(model, free_blocks_chk, clamped_blocks=[])

    # Random-halves program
    rng = np.random.RandomState(123)
    free_blocks_rand = build_random_blocks(flat_nodes, rng)
    prog_rand = IsingSamplingProgram(model, free_blocks_rand, clamped_blocks=[])

    thr_chk, m_chk   = eff_samples_per_sec(model, prog_chk,  flat_nodes, free_blocks_chk,  steps=128, sps=1, seed=0)
    thr_rand, m_rand = eff_samples_per_sec(model, prog_rand, flat_nodes, free_blocks_rand, steps=128, sps=1, seed=1)

    a1_chk  = lag1_autocorr(m_chk)
    a1_rand = lag1_autocorr(m_rand)

    print("Checkerboard  -> samples/sec: %.1f | lag1 autocorr: %.3f" % (thr_chk,  a1_chk))
    print("Random halves -> samples/sec: %.1f | lag1 autocorr: %.3f" % (thr_rand, a1_rand))

    with open("outputs/ablations.txt","w") as f:
        f.write(f"checkerboard_samples_per_sec={thr_chk:.2f}\n")
        f.write(f"checkerboard_lag1_autocorr={a1_chk:.3f}\n")
        f.write(f"random_samples_per_sec={thr_rand:.2f}\n")
        f.write(f"random_lag1_autocorr={a1_rand:.3f}\n")
    print("Saved outputs/ablations.txt")


if __name__ == "__main__":
    main()
