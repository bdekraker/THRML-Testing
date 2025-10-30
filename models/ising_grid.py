import jax.numpy as jnp
from thrml import Block
from thrml.pgm import SpinNode
from thrml.models.ising import IsingEBM, IsingSamplingProgram

def make_grid_ising(H=28, W=28, J=0.5, beta=1.0):
    nodes = [[SpinNode() for _ in range(W)] for _ in range(H)]
    flat = [n for row in nodes for n in row]
    edges = []
    for i in range(H):
        for j in range(W):
            if i+1 < H: edges.append((nodes[i][j], nodes[i+1][j]))
            if j+1 < W: edges.append((nodes[i][j], nodes[i][j+1]))
    biases  = jnp.zeros((H*W,), jnp.float32)
    weights = jnp.ones((len(edges),), jnp.float32) * J
    beta    = jnp.array(beta, jnp.float32)
    model = IsingEBM(flat, edges, biases, weights, beta)

    evens, odds = [], []
    for i in range(H):
        for j in range(W):
            (evens if (i+j)%2==0 else odds).append(nodes[i][j])
    free_blocks = [Block(evens), Block(odds)]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    return model, free_blocks, program, flat
