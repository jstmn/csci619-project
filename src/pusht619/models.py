from __future__ import annotations

from time import time

PROGRAM_START_TIME = time()
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from termcolor import cprint
import jax

jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache/jax_pusht619"))
import jax.numpy as jnp
import numpy as np
import argparse
import gurobipy as gp
from gurobipy import GRB

from pusht619.core import PushTEnv, ANGLE_BOUNDS, CONTACT_POINT_BOUNDS


# ── Hyperparameters ───────────────────────────────────────────────────────────

N_OPT_STEPS = 1000
LR = 0.1
N_SIM_STEPS = 50
RESET_SEED = 0
N_FACES = 6
RANDOMZED_SMOOTHING_K = 10
OUTPUT_REG_BETA = 1e-3
CP_TARGET_WEIGHT = 1.0
ANG_TARGET_WEIGHT = 1.0

# Randomized smoothing scale: perturbed costs are c + λ ε, ε ~ N(0, I).
# Too small → perturbed solves often match x*; estimator variance high.
# Too large → x_k far from x*; gradient bias grows.
PERTURB_LAMBDA = 0.1


# ── Gurobi action solver ──────────────────────────────────────────────────────

_lo_cp, _hi_cp = CONTACT_POINT_BOUNDS
_lo_ang, _hi_ang = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])
_mid_cp = 0.5 * (_lo_cp + _hi_cp)
_mid_ang = 0.5 * (_lo_ang + _hi_ang)



# ── MLP: y → c ────────────────────────────────────────────────────────────────

class MLP:
    """Maps context y (dim=11) → solver parameters c (dim=8).

    c[:6] = face cost coefficients
    c[6]  = contact_point target, squashed into CONTACT_POINT_BOUNDS
    c[7]  = angle target, squashed into ANGLE_BOUNDS

    Params: list of (W, b) tuples — plain JAX pytree, jit/grad compatible.
    """

    def __init__(self, context_dim: int, hidden_dims: Sequence[int] = (128, 128)):
        self.layer_sizes = [context_dim, *hidden_dims, 8]
        self.cp_bounds = (_lo_cp, _hi_cp)
        self.ang_bounds = (_lo_ang, _hi_ang)

    def init(self, key: jax.Array) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        params = []
        for i in range(len(self.layer_sizes) - 1):
            key, subkey = jax.random.split(key)
            fan_in, fan_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            w = jax.random.normal(subkey, (fan_in, fan_out), dtype=jnp.float32) * jnp.sqrt(2.0 / fan_in)
            b = jnp.zeros(fan_out, dtype=jnp.float32)
            params.append((w, b))
        return params

    def apply(self, params: list[tuple[jnp.ndarray, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass. Returns face logits plus bounded continuous targets."""
        for i, (w, b) in enumerate(params):
            x = x @ w + b
            if i < len(params) - 1:
                x = jax.nn.relu(x)
        face_logits = x[:, :N_FACES]
        cp_lo, cp_hi = self.cp_bounds
        ang_lo, ang_hi = self.ang_bounds
        cp_target = cp_lo + (cp_hi - cp_lo) * jax.nn.sigmoid(x[:, 6:7])
        ang_target = ang_lo + (ang_hi - ang_lo) * jax.nn.sigmoid(x[:, 7:8])
        return jnp.concatenate([face_logits, cp_target, ang_target], axis=-1)

    def save_mlp_weights(self, filepath: Path, params: list[tuple[jnp.ndarray, jnp.ndarray]]) -> Path:
        """Save MLP weights, biases, and output limits for one training iteration."""
        cp_lo, cp_hi = self.cp_bounds
        ang_lo, ang_hi = self.ang_bounds
        arrays = {
            "cp_bounds": np.asarray([cp_lo, cp_hi], dtype=np.float32),
            "ang_bounds": np.asarray([ang_lo, ang_hi], dtype=np.float32),
            "layer_sizes": np.asarray(self.layer_sizes, dtype=np.int32),
        }
        for layer_idx, (w, b) in enumerate(params):
            arrays[f"layer_{layer_idx}_w"] = np.asarray(w)
            arrays[f"layer_{layer_idx}_b"] = np.asarray(b)
        np.savez(filepath, **arrays)
