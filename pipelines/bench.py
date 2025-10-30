#!/usr/bin/env python3
import argparse, json, time, os, numpy as np, jax
from thrml import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import hinton_init, IsingSamplingProgram
from models.ising_grid import make_grid_ising


def run_once(H=28, W=28, J=0.35, warmup=100, n_samples=128, sps=1, seed=0,
             blocking="checkerboard", out_json="outputs/results.json"):
    # Build model
    model, free_blocks_chk, _, flat = make_grid_ising(H, W, J=J)

    # Choose blocking
    if blocking == "checkerboard":
        free_blocks = free_blocks_chk
    elif blocking == "random":
        rng = np.random.default_rng(123)
        idx = np.arange(len(flat))
        rng.shuffle(idx)
        half = len(idx)//2
        b1 = [flat[i] for i in idx[:half]]
        b2 = [flat[i] for i in idx[half:]]
        free_blocks = [Block(b1), Block(b2)]
    else:
        raise ValueError("blocking must be 'checkerboard' or 'random'")

    program = IsingSamplingProgram(model, free_blocks=free_blocks, clamped_blocks=[])
    key = jax.random.key(seed)
    init = hinton_init(key, model, free_blocks, batch_shape=())
    sched = SamplingSchedule(n_warmup=warmup, n_samples=n_samples, steps_per_sample=sps)

    t0 = time.time()
    samples = sample_states(key, program, sched, init_state_free=init, state_clamp=[],
                            nodes_to_sample=[Block(flat)])[0]
    dt = time.time() - t0

    # magnetization series + simple mixing proxy (lag-1 autocorr)
    m = np.array(samples.mean(axis=1))
    m0 = m - m.mean()
    lag1 = float(np.dot(m0[:-1], m0[1:]) / np.dot(m0, m0))

    res = {
        "device": [str(d) for d in jax.devices()],
        "H": H, "W": W, "J": J,
        "warmup": warmup, "n_samples": n_samples, "steps_per_sample": sps,
        "blocking": blocking, "seed": seed,
        "wall_sec": dt, "samples_per_sec": n_samples / dt,
        "lag1_autocorr": lag1,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=28)
    p.add_argument("--W", type=int, default=28)
    p.add_argument("--J", type=float, default=0.35)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--sps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blocking", choices=["checkerboard","random"], default="checkerboard")
    p.add_argument("--out", default="outputs/results.json")
    args = p.parse_args()
    run_once(args.H, args.W, args.J, args.warmup, args.n_samples, args.sps,
             args.seed, args.blocking, args.out)
