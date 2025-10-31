# RunPod / GPU quickstart

This folder contains a CUDA-ready Docker image so you can reproduce the THRML sweeps on cloud GPUs (RunPod, vast.ai, AWS g5, etc.).

## 1. Build the image (one-time per GPU host)

```bash
docker build -t thrml:cuda12 -f docker/Dockerfile.gpu .
```

The image includes:

- Python + CUDA 12 runtime
- `jax[cuda12_pip]` wheels
- THRML + helper libraries (`numpy`, `pillow`, `matplotlib`, `seaborn`, …)

## 2. Launch on RunPod

1. Create a container (e.g. 1×4090 or 1×L40S) with Docker runtime enabled.
2. SSH into the pod and clone this repository:

   ```bash
   git clone https://github.com/bdekraker/THRML-Testing.git
   cd THRML-Testing
   ```

3. Build and reuse the image if needed (`docker build …`).
4. Start an interactive session mounting the repo:

   ```bash
   docker run --gpus all -it --rm \
     -v $PWD:/workspace \
     thrml:cuda12 bash
   ```

Inside the container `/workspace` contains the repo. You can now run any pipeline with CUDA acceleration.

## 3. Example commands inside the container

```bash
cd /workspace
export PYTHONPATH=/workspace

# Schedule & blocking sweep (GPU)
python pipelines/bench.py \
  --blocking checkerboard stripes supercell random \
  --J 0.30 0.45 0.60 \
  --sps 4 6 \
  --num_chains 4 8 \
  --seed 0 \
  --out outputs/bench_gpu_hidden_gold.json

# Mixing curves for GPU run
python viz/autocorr_magnetization.py \
  --sps 6 \
  --blockings checkerboard stripes supercell random \
  --J 0.30 \
  --steps 400 \
  --maxlag 50 \
  --seed 0

# Benchmark plots
python viz/plot_bench.py
```

The helper script `docker/run_bench.sh` wraps the common flags; make it executable (`chmod +x docker/run_bench.sh`) and run:

```bash
./docker/run_bench.sh
```

Environment variables (`BLOCKING`, `STEPS_PER_SAMPLE`, `NUM_CHAINS`, `J_VALUES`, `SEED`, `OUT`) let you customise without editing the script.

## 4. Publishing back to the scoreboard

1. Copy `outputs/bench_*.json` and figures off the pod (`scp`, `rsync`, or Git).
2. Create a new JSON under `THRML-Leaderboard/results/` (e.g. `l4_checkerboard_sps6.json`) with the GPU metrics.
3. Run `python scripts/validate_results.py && python scripts/build_scoreboard.py` locally.
4. Commit and push; the GitHub Action redeploys the leaderboard automatically.

That’s it—you’re now ready to generate GPU baselines that slot straight into the scoreboard. Happy hunting for hidden gold!
