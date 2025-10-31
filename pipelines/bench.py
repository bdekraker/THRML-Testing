#!/usr/bin/env python3
import argparse
import json
import os
import time
from itertools import product

import jax
import numpy as np
from thrml import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import IsingSamplingProgram, hinton_init
from models.ising_grid import make_grid_ising


def ar1_tau_int(rho1: float) -> float:
    """AR(1) integrated autocorrelation time approximation."""
    rho1 = float(np.clip(rho1, -0.999, 0.999))
    return (1.0 + rho1) / (1.0 - rho1)


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
    raise ValueError("blocking must be one of 'checkerboard', 'random', 'stripes', 'supercell'")


def run_once(
    H=28,
    W=28,
    J=0.35,
    warmup=100,
    n_samples=128,
    sps=1,
    seed=0,
    num_chains=1,
    blocking="checkerboard",
):
    model, free_blocks_chk, _, flat_nodes = make_grid_ising(H, W, J=J)
    free_blocks = build_free_blocks(flat_nodes, free_blocks_chk, blocking, seed, H, W)
    program = IsingSamplingProgram(model, free_blocks=free_blocks, clamped_blocks=[])
    schedule = SamplingSchedule(
        n_warmup=warmup, n_samples=n_samples, steps_per_sample=sps
    )

    base_key = jax.random.key(seed)
    chain_keys = jax.random.split(base_key, num_chains)

    def single_chain(chain_key):
        init_key, sample_key = jax.random.split(chain_key)
        init_state = hinton_init(init_key, model, free_blocks, batch_shape=())
        samples = sample_states(
            sample_key,
            program,
            schedule,
            init_state_free=init_state,
            state_clamp=[],
            nodes_to_sample=[Block(flat_nodes)],
        )[0]
        return samples

    if num_chains == 1:
        t0 = time.time()
        samples = single_chain(chain_keys[0])
        samples = jax.device_get(samples)
        dt = time.time() - t0
        samples = np.expand_dims(samples, axis=0)
    else:
        batched = jax.vmap(single_chain)
        t0 = time.time()
        samples = batched(chain_keys)
        samples = jax.device_get(samples)
        dt = time.time() - t0

    samples = np.array(samples)
    if samples.ndim == 2:
        samples = samples[None, ...]
    chains, sample_steps, _ = samples.shape

    magnetizations = samples.mean(axis=2)
    centered = magnetizations - magnetizations.mean(axis=1, keepdims=True)
    numer = np.sum(centered[:, :-1] * centered[:, 1:], axis=1)
    denom = np.sum(centered**2, axis=1)
    denom = np.where(denom > 0, denom, 1.0)
    lag1_per_chain = numer / denom
    tau_per_chain = np.array([ar1_tau_int(r) for r in lag1_per_chain])
    ess_per_chain = np.where(tau_per_chain > 0, sample_steps / tau_per_chain, 0.0)

    total_samples = sample_steps * chains
    wall = float(dt)
    samples_per_sec = float(total_samples / wall) if wall > 0 else float("inf")
    ess_per_sec = float(ess_per_chain.sum() / wall) if wall > 0 else 0.0

    result = {
        "device": [str(d) for d in jax.devices()],
        "H": H,
        "W": W,
        "J": float(J),
        "warmup": warmup,
        "n_samples": n_samples,
        "steps_per_sample": sps,
        "blocking": blocking,
        "seed": seed,
        "num_chains": chains,
        "wall_sec": wall,
        "samples_per_sec": samples_per_sec,
        "lag1_autocorr": float(lag1_per_chain.mean()),
        "lag1_autocorr_per_chain": lag1_per_chain.tolist(),
        "tau_int_est": float(tau_per_chain.mean()),
        "tau_int_est_per_chain": tau_per_chain.tolist(),
        "ess_per_sec": ess_per_sec,
        "ess_per_sec_per_chain": (ess_per_chain / wall).tolist()
        if wall > 0
        else [0.0] * chains,
        "samples_per_sec_per_chain": float(sample_steps / wall)
        if wall > 0
        else float("inf"),
        "total_samples": total_samples,
    }
    return result


def ensure_directory(path):
    if path.endswith(".json"):
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=28)
    parser.add_argument("--W", type=int, default=28)
    parser.add_argument("--J", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--sps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--blocking",
        choices=["checkerboard", "random", "stripes", "supercell"],
        nargs="+",
        default=["checkerboard", "stripes", "supercell", "random"]
    )
    parser.add_argument("--num_chains", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    combos = list(
        product(args.blocking, args.J, args.sps, args.num_chains)
    )
    ensure_directory(args.out)
    multi = len(combos) > 1
    out_is_file = args.out.endswith(".json")
    if out_is_file:
        out_dir = os.path.dirname(args.out) or "."
        base_name = os.path.splitext(os.path.basename(args.out))[0]
    else:
        out_dir = args.out
        base_name = "bench"
    os.makedirs(out_dir, exist_ok=True)

    aggregate = []
    for blocking, J_val, sps_val, n_chains in combos:
        res = run_once(
            H=args.H,
            W=args.W,
            J=J_val,
            warmup=args.warmup,
            n_samples=args.n_samples,
            sps=sps_val,
            seed=args.seed,
            num_chains=n_chains,
            blocking=blocking,
        )
        aggregate.append(res)

        if not multi and out_is_file:
            out_path = args.out
        else:
            out_path = os.path.join(
                out_dir,
                f"{base_name}_{blocking}_J{J_val:.2f}_sps{sps_val}_chains{n_chains}.json",
            )
        res["_out_path"] = out_path
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        print(f"Saved {out_path}")

    if multi and out_is_file:
        with open(args.out, "w") as f:
            json.dump(aggregate, f, indent=2)
        print(f"Wrote aggregate results to {args.out}")


if __name__ == "__main__":
    main()
