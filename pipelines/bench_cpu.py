import time
import jax
from thrml import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import hinton_init
from models.ising_grid import make_grid_ising

def bench(H=28, W=28, warmup=100, n_samples=64, sps=1):
    print("Devices:", jax.devices())
    model, free_blocks, program, flat_nodes = make_grid_ising(H,W)
    key = jax.random.key(0)
    init = hinton_init(key, model, free_blocks, batch_shape=())
    schedule = SamplingSchedule(n_warmup=warmup, n_samples=n_samples, steps_per_sample=sps)
    t0=time.time()
    _ = sample_states(key, program, schedule, init_state_free=init, state_clamp=[], nodes_to_sample=[Block(flat_nodes)])
    total = time.time()-t0
    eff = n_samples/total
    print(f"CPU bench: {H}x{W}, warmup={warmup}, n_samples={n_samples}, sps={sps}")
    print(f"Wall: {total:.2f}s | Samples/s: {eff:.1f}")

if __name__=="__main__":
    bench()
