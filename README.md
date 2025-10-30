# THRML Speedrun (CPU → GPU)

Small, reproducible demos using [THRML] + JAX:
- Ising sampler (samples + mixing GIF)
- Mixing curves (sps tradeoff)
- Block ablation (checkerboard vs random)
- Conditional sampling (inpainting)

Run from repo root:
```bash
python -m pipelines.sample_ising
python -m viz.autocorr_magnetization
python -m pipelines.ablations
python -m pipelines.inpaint_ising


