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

from pusht619.core import ANGLE_BOUNDS, CONTACT_POINT_BOUNDS, NUM_FACES


# ── Hyperparameters ───────────────────────────────────────────────────────────

N_OPT_STEPS = 1000
LR = 0.1
N_SIM_STEPS = 50
RESET_SEED = 0
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
    """Maps context y → solver parameters c.

    Each 8-D output block stores:

    c[:6]  = face cost coefficients (logits)
    c[6]   = shared contact_point target, squashed into CONTACT_POINT_BOUNDS
    c[7]   = shared angle target, squashed into ANGLE_BOUNDS

    ``output_dim`` may contain one block (single-step) or multiple consecutive
    blocks (multi-step). Params: list of (W, b) tuples — plain JAX pytree,
    jit/grad compatible.
    """

    def __init__(self, context_dim: int, hidden_dims: Sequence[int] = (128, 128), output_dim: int = NUM_FACES + 2):
        self.layer_sizes = [context_dim, *hidden_dims, output_dim]
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
        """Forward pass. Returns bounded solver parameters in 8-D blocks."""
        for i, (w, b) in enumerate(params):
            x = x @ w + b
            if i < len(params) - 1:
                x = jax.nn.relu(x)
        if x.shape[-1] % (NUM_FACES + 2) != 0:
            raise ValueError(f"output_dim must be divisible by {NUM_FACES + 2}, got {x.shape[-1]}")
        x = x.reshape(x.shape[0], -1, NUM_FACES + 2)
        face_logits = x[:, :, :NUM_FACES]
        cp_lo, cp_hi = self.cp_bounds
        ang_lo, ang_hi = self.ang_bounds
        cp_target = cp_lo + (cp_hi - cp_lo) * jax.nn.sigmoid(x[:, :, NUM_FACES : NUM_FACES + 1])
        ang_target = ang_lo + (ang_hi - ang_lo) * jax.nn.sigmoid(x[:, :, NUM_FACES + 1 : NUM_FACES + 2])
        out = jnp.concatenate([face_logits, cp_target, ang_target], axis=-1)
        return out.reshape(out.shape[0], -1)

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


class ActionSolver:
    """Gurobi solver matching the training-time objective."""

    def __init__(self):
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.model = gp.Model(env=self.env)
        self.model.setParam("Threads", 1)
        self.model.setParam("Presolve", 0)
        self.xf = self.model.addVars(NUM_FACES, vtype=GRB.BINARY, name="xf")
        self.cp = self.model.addVar(lb=_lo_cp, ub=_hi_cp, vtype=GRB.CONTINUOUS, name="cp")
        self.ang = self.model.addVar(lb=_lo_ang, ub=_hi_ang, vtype=GRB.CONTINUOUS, name="ang")
        self.model.addConstr(gp.quicksum(self.xf[i] for i in range(NUM_FACES)) == 1)
        self.model.update()

    def solve(self, c: np.ndarray) -> np.ndarray:
        """Update objective from c (8,), solve, return x (8,)."""
        if not np.all(np.isfinite(c)):
            c = np.zeros(NUM_FACES + 2, dtype=np.float32)
            c[NUM_FACES] = _mid_cp
            c[NUM_FACES + 1] = _mid_ang
        cp_ref = float(np.clip(c[NUM_FACES], _lo_cp, _hi_cp))
        ang_ref = float(np.clip(c[NUM_FACES + 1], _lo_ang, _hi_ang))
        face_obj = gp.quicksum(float(c[i]) * self.xf[i] for i in range(NUM_FACES))
        cp_obj = CP_TARGET_WEIGHT * (self.cp - cp_ref) * (self.cp - cp_ref)
        ang_obj = ANG_TARGET_WEIGHT * (self.ang - ang_ref) * (self.ang - ang_ref)
        self.model.setObjective(face_obj + cp_obj + ang_obj, GRB.MINIMIZE)
        self.model.update()
        self.model.optimize()
        face_vals = np.array([self.xf[i].X for i in range(NUM_FACES)], dtype=np.float32)
        x = np.append(face_vals, [self.cp.X, self.ang.X]).astype(np.float32)
        assert x.shape == (NUM_FACES + 2,), f"x must be ({NUM_FACES + 2},), got {x.shape}"
        return x

    def solve_batch(self, c_batch: np.ndarray) -> np.ndarray:
        """Solve the Gurobi objective for every env. Returns (N, {NUM_FACES + 2})."""
        out = np.zeros((c_batch.shape[0], NUM_FACES + 2), dtype=np.float32)
        for i in range(c_batch.shape[0]):
            out[i] = self.solve(c_batch[i])
        return out


class ActionSolverMultiStep:
    """MIQP over a fixed horizon of M consecutive actions.

    Each action m = 0 … M-1 has the same structure as :class:`ActionSolver`, but
    the six face variables are *independent* binaries (no one-hot / no sum
    constraint). The total objective is the sum of the M per-step objectives.

    - ``c`` layout: ``(8 * M,)`` with blocks ``c[8*m : 8*(m+1)]`` =
      ``[face linear costs (6), cp target, ang target]`` for step ``m``.

    - ``x`` layout: ``(8 * M,)`` with blocks ``[face bits (6), cp, ang]`` per step.
    """

    def __init__(self, n_actions: int = 3):
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1")
        self.n_actions = int(n_actions)
        self.action_dim = NUM_FACES + 2
        self.c_dim = self.action_dim * self.n_actions

        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()
        self.model = gp.Model(env=self.env)
        self.model.setParam("Threads", 1)
        self.model.setParam("Presolve", 0)

        self.xf: list = []
        self.cp: list = []
        self.ang: list = []
        for m in range(self.n_actions):
            self.xf.append(self.model.addVars(NUM_FACES, vtype=GRB.BINARY, name=f"xf_{m}"))
            self.cp.append(self.model.addVar(lb=_lo_cp, ub=_hi_cp, vtype=GRB.CONTINUOUS, name=f"cp_{m}"))
            self.ang.append(self.model.addVar(lb=_lo_ang, ub=_hi_ang, vtype=GRB.CONTINUOUS, name=f"ang_{m}"))
        self.model.update()

    def solve(self, c: np.ndarray) -> np.ndarray:
        """Update objective from c ((NUM_FACES + 2) * n_actions,), solve, return x ((NUM_FACES + 2) * n_actions,)."""
        c = np.asarray(c, dtype=np.float64).ravel()
        if c.shape != (self.c_dim,):
            raise ValueError(f"c must have shape ({self.c_dim},), got {c.shape}")
        if not np.all(np.isfinite(c)):
            c = np.zeros(self.c_dim, dtype=np.float32)
            for m in range(self.n_actions):
                c[(NUM_FACES + 2) * m + NUM_FACES] = _mid_cp
                c[(NUM_FACES + 2) * m + NUM_FACES + 1] = _mid_ang

        terms = []
        for m in range(self.n_actions):
            lo = (NUM_FACES + 2) * m
            cm = c[lo : lo + (NUM_FACES + 2)]
            cp_ref = float(np.clip(cm[NUM_FACES], _lo_cp, _hi_cp))
            ang_ref = float(np.clip(cm[NUM_FACES + 1], _lo_ang, _hi_ang))
            face_obj = gp.quicksum(float(cm[i]) * self.xf[m][i] for i in range(NUM_FACES))
            cp_obj = CP_TARGET_WEIGHT * (self.cp[m] - cp_ref) * (self.cp[m] - cp_ref)
            ang_obj = ANG_TARGET_WEIGHT * (self.ang[m] - ang_ref) * (self.ang[m] - ang_ref)
            terms.append(face_obj + cp_obj + ang_obj)
        obj = terms[0]
        for t in terms[1:]:
            obj = obj + t
        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.update()
        self.model.optimize()

        parts: list[np.ndarray] = []
        for m in range(self.n_actions):
            face_vals = np.array([self.xf[m][i].X for i in range(NUM_FACES)], dtype=np.float32)
            parts.append(np.append(face_vals, [self.cp[m].X, self.ang[m].X]).astype(np.float32))
        x = np.concatenate(parts, axis=0)
        assert x.shape == (self.c_dim,), f"x must be ({self.c_dim},), got {x.shape}"
        return x

    def solve_batch(self, c_batch: np.ndarray) -> np.ndarray:
        """Solve the Gurobi objective for every env. Returns (N, (NUM_FACES + 2) * self.n_actions)."""
        c_batch = np.asarray(c_batch)
        if c_batch.ndim != 2 or c_batch.shape[1] != self.c_dim:
            raise ValueError(f"c_batch must be (N, (NUM_FACES + 2) * n_actions), got {c_batch.shape}")
        n = c_batch.shape[0]
        out = np.zeros((n, (NUM_FACES + 2) * self.n_actions), dtype=np.float32)
        for i in range(n):
            out[i] = self.solve(c_batch[i])
        return out
