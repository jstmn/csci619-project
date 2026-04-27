"""
Push-T parallel environment.

Block state: (x, y, theta) where (x, y) is the link-frame origin in world
coordinates and theta is the CCW rotation in radians.

Faces (A-F) go around the perimeter of the T; corners p0-p7 are at floor
level (z = 0) in the body frame defined in the URDF / README.
"""

from pathlib import Path
import copy
from dataclasses import dataclass
import functools
from typing import Any

import numpy as np
from lxml import etree
import imageio
import jax
import jax.numpy as jnp
import jaxsim.api as js
import jaxsim.mujoco
import jaxsim.rbda.contacts as _jsc
import matplotlib.pyplot as plt
import mujoco as mj
import rod


# Action bounds (enforced on concrete :class:`Action` and checked every rollout in ``_step_pure_impl``).
CONTACT_POINT_BOUNDS = (0.2, 0.8)
ANGLE_BOUNDS = (jnp.pi * 0.25, jnp.pi * 0.75)


def _raise_unless_contact_angle_bounds(cp: np.ndarray, ang: np.ndarray) -> None:
    """Host callback: raises ``ValueError`` if any env has non-finite or out-of-range values."""
    lo_cp, hi_cp = CONTACT_POINT_BOUNDS
    lo_ang, hi_ang = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])
    cp = np.asarray(cp, dtype=np.float32).reshape(-1)
    ang = np.asarray(ang, dtype=np.float32).reshape(-1)
    if cp.shape != ang.shape:
        raise ValueError(f"contact_point and angle length mismatch: {cp.shape} vs {ang.shape}")
    for name, arr, lo, hi in (
        ("contact_point", cp, lo_cp, hi_cp),
        ("angle", ang, lo_ang, hi_ang),
    ):
        ok = np.isfinite(arr) & (arr >= lo) & (arr <= hi)
        bad = np.where(~ok)[0]
        if len(bad):
            raise ValueError(
                f"{name} must be finite and in [{lo}, {hi}]; bad env indices {bad.tolist()} values {arr[bad].tolist()}"
            )


T_RADIUS = 0.15
WORKSPACE_WIDTH = 1.5
WORKSPACE_HEIGHT = 1.5
PUSHER_RADIUS = 0.01
PUSHER_CLEARANCE = 0.005
PUSHER_APPROACH_DISTANCE = 0.005  # 0.04
DEFAULT_PUSH_DISTANCE = 0.30
PUSHER_KP = 400.0
PUSHER_KD = 40.0
PUSHER_MAX_FORCE = 80.0

# Target speeds for the two trajectory phases of a single step().
# The approach phase moves the pusher from `pusher_start` to `pre_contact`
# (`PUSHER_APPROACH_DISTANCE`) at `PUSHER_APPROACH_VELOCITY`; the push phase
# then advances `DEFAULT_PUSH_DISTANCE` at `PUSHER_PUSH_VELOCITY`.  Any leftover
PUSHER_APPROACH_VELOCITY = 8.0  # 4.0 # m/s
PUSHER_PUSH_VELOCITY = 1.0  # m/s

# JaxSim only supports point-vs-terrain contacts, not body-vs-body, so we add
# our own analytical spring-damper contact between the pusher and the T block.
T_PUSHER_CONTACT_K = 2.0e4  # N/m  (~1 mm penetration at 20 N)
T_PUSHER_CONTACT_D = 80.0  # N·s/m
PUSHER_EFFECTIVE_RADIUS = PUSHER_RADIUS  # collision radius of the pusher

# Spring-damper wall contacts: each of the 8 T corners is checked against the
# 4 axis-aligned workspace walls.  For flat half-plane walls, corner checking
# is equivalent to edge checking (deepest point on any segment is always an endpoint).
T_WALL_CONTACT_K = 2.0e4  # N/m
T_WALL_CONTACT_D = 80.0  # N·s/m

# Explicit joint-space friction on the T<->floor interface. We disable JaxSim's
# implicit terrain tangential friction (contact mu=0) and model the floor
# friction here so the behaviour is fully controllable.
# Total effective friction force on a moving T translating with velocity v is:
#     F_fric = -(T_FLOOR_MU * T_MASS * G) * tanh(v / T_FLOOR_V_EPS) - T_FLOOR_C_LIN * v
# and similarly for rotation, with T_FLOOR_ROT_RADIUS giving the moment arm
# used for Coulomb rotational friction.
T_MASS = 0.6  # kg, matches the SDF <mass> on link t_block
_GRAVITY_MAG = 9.81
FRICTION_SCALE = 6.0
T_FLOOR_MU = FRICTION_SCALE * 0.35  # Coulomb coefficient
T_FLOOR_C_LIN = FRICTION_SCALE * 2.0  # N·s/m  (viscous linear)
T_FLOOR_C_ROT = FRICTION_SCALE * 0.02  # N·m·s/rad  (viscous rotational)
T_FLOOR_V_EPS = 5.0e-3  # m/s   (tanh smoothing: below this, behaves like viscous)
T_FLOOR_W_EPS = 5.0e-2  # rad/s (tanh smoothing for rotational)
T_FLOOR_ROT_RADIUS = FRICTION_SCALE * 0.086  # m, ≈ sqrt(izz/m) from the SDF (4.469e-3 / 0.6)

# T body-frame 2D geometry (xy only; z is ignored because everything lives on the floor).
# Matches the two collision boxes defined in assets/scene.sdf for link "t_block".
T_TOP_BAR_CENTER = jnp.array([0.0, 0.10])
T_TOP_BAR_HALF = jnp.array([0.10, 0.025])
T_STEM_CENTER = jnp.array([0.0, -0.025])
T_STEM_HALF = jnp.array([0.025, 0.10])

T_CORNERS = np.array(
    [
        [-0.025, -0.125],  # p0
        [-0.025, 0.075],  # p1
        [-0.10, 0.075],  # p2
        [-0.10, 0.125],  # p3
        [0.10, 0.125],  # p4
        [0.10, 0.075],  # p5
        [0.025, 0.075],  # p6
        [0.025, -0.125],  # p7
    ],
    dtype=np.float32,
)
# Restrict contact to the A/E/C faces by aliasing the 6 face slots onto only
# those three physical edges. Per the READM
FACE_START_POINTS = T_CORNERS[[0, 7, 6, 3]]
FACE_END_POINTS = T_CORNERS[[1, 0, 7, 4]]
NUM_FACES = len(FACE_START_POINTS)
# FACE_START_POINTS = T_CORNERS[[0, 2, 3, 4, 6, 7]]
# FACE_END_POINTS = T_CORNERS[[1, 3, 4, 5, 7, 0]]
_FACE_START_POINTS_JAX = jnp.asarray(FACE_START_POINTS)
_FACE_END_POINTS_JAX = jnp.asarray(FACE_END_POINTS)
_T_CORNERS_JAX = jnp.asarray(T_CORNERS)  # (8, 2) body-frame corners
_DIST_CORNERS_JAX = jnp.asarray(T_CORNERS[[0, 3, 4, 7]])  # (4, 2) p0, p3, p4, p7 used for the distance metric


def _plan_push_jax(
    t_poses: jnp.ndarray,  # (nenvs, 3)  [x, y, theta]
    face: jnp.ndarray,  # (nenvs,) int
    contact_point: jnp.ndarray,  # (nenvs,)  must lie in CONTACT_POINT_BOUNDS
    angle: jnp.ndarray,  # (nenvs,)  must lie in ANGLE_BOUNDS
    nenvs: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX re-implementation of `_plan_push`.

    Pure and differentiable w.r.t. `contact_point`, `angle`, `t_poses`, and —
    when `face` is passed as a `(nenvs, NUM_FACES)` float array of per-face weights —
    also w.r.t. `face`. A `(nenvs,)` int `face` behaves like a hard gather and
    carries zero gradient.
    """
    assert t_poses.shape == (nenvs, 3), f"t_poses must be (nenvs, 3), got {t_poses.shape}"
    assert face.shape in ((nenvs,), (nenvs, NUM_FACES)), (
        f"face must be (nenvs,) or (nenvs, NUM_FACES), got {face.shape}"
    )
    assert contact_point.shape == (nenvs,), f"contact_point must be (nenvs,), got {contact_point.shape}"
    assert angle.shape == (nenvs,), f"angle must be (nenvs,), got {angle.shape}"

    # --- Face geometry (body frame) ---
    # Hard mode (int): index directly into the precomputed table — no gradient through face.
    # Soft mode (float weights): take a convex combination of all face endpoints.
    #   face_starts/ends become a weighted average of every face's start/end point,
    #   so gradients flow back into face_weights through the matmul.
    if face.ndim == 1:
        face_starts = _FACE_START_POINTS_JAX[face]  # (nenvs, 2) — hard gather, no gradient
        face_ends = _FACE_END_POINTS_JAX[face]
    else:
        # (nenvs, NUM_FACES) @ (NUM_FACES, 2) → (nenvs, 2): differentiable weighted average
        face_starts = face @ _FACE_START_POINTS_JAX
        face_ends = face @ _FACE_END_POINTS_JAX

    # Build a local coordinate frame for the (blended) face edge:
    #   tangent  — unit vector along the face edge
    #   inward normal — 90° CCW from tangent, pointing into the T block
    face_vectors = face_ends - face_starts
    face_lengths = jnp.linalg.norm(face_vectors, axis=1, keepdims=True)
    face_tangents = face_vectors / face_lengths
    face_inward_normals = jnp.stack([face_tangents[:, 1], -face_tangents[:, 0]], axis=-1)

    # Place the contact point along the face edge using the scalar contact_point parameter.
    # Push direction is a blend of tangent and inward normal controlled by `angle`:
    #   angle=π/2 → purely inward (perpendicular to face), angle=0 → along the edge.
    contact_body = face_starts + contact_point[:, None] * face_vectors
    push_direction_body = jnp.cos(angle)[:, None] * face_tangents + jnp.sin(angle)[:, None] * face_inward_normals

    # --- Body → world transform ---
    # Rotate contact point and push direction from the T block's local frame
    # into the world frame using the T block's current orientation theta.
    cos_theta = jnp.cos(t_poses[:, 2])
    sin_theta = jnp.sin(t_poses[:, 2])
    rotation_matrices = jnp.stack(
        [
            jnp.stack([cos_theta, -sin_theta], axis=-1),
            jnp.stack([sin_theta, cos_theta], axis=-1),
        ],
        axis=1,
    )
    assert t_poses.shape[1:] == (3,), f"t_poses must be (nenvs, 3), got {t_poses.shape}"
    assert rotation_matrices.shape[1:] == (2, 2), (
        f"rotation_matrices must be (nenvs, 2, 2), got {rotation_matrices.shape}"
    )
    assert contact_body.shape[1:] == (2,), f"contact_body must be (nenvs, 2), got {contact_body.shape}"
    assert push_direction_body.shape[1:] == (2,), (
        f"push_direction_body must be (nenvs, 2), got {push_direction_body.shape}"
    )
    contact_world = t_poses[:, :2] + jnp.einsum("nij,nj->ni", rotation_matrices, contact_body)
    push_direction_world = jnp.einsum("nij,nj->ni", rotation_matrices, push_direction_body)
    assert contact_world.shape[1:] == (2,), f"contact_world must be (nenvs, 2), got {contact_world.shape}"
    assert push_direction_world.shape[1:] == (2,), (
        f"push_direction_world must be (nenvs, 2), got {push_direction_world.shape}"
    )
    push_direction_world = push_direction_world / jnp.linalg.norm(push_direction_world, axis=1, keepdims=True)

    # --- Pusher placement ---
    # pre_contact: pusher surface just touching the T block (offset by pusher radius + clearance).
    # pusher_start: approach position further back along the push direction.
    pre_contact = contact_world - (PUSHER_RADIUS + PUSHER_CLEARANCE) * push_direction_world
    pusher_start = pre_contact - PUSHER_APPROACH_DISTANCE * push_direction_world

    workspace_margin = PUSHER_RADIUS
    lb = jnp.array([workspace_margin, workspace_margin])
    ub = jnp.array([WORKSPACE_WIDTH - workspace_margin, WORKSPACE_HEIGHT - workspace_margin])
    pusher_start = jnp.clip(pusher_start, lb, ub)
    pre_contact = jnp.clip(pre_contact, lb, ub)
    return pusher_start, pre_contact, push_direction_world


def _box_sdf_2d(p: jnp.ndarray, c: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """Signed distance from `p` to an axis-aligned 2D box centered at `c` with half-extents `h`.

    Negative inside, positive outside; differentiable almost everywhere.
    """
    q = jnp.abs(p - c) - h
    outside_q = jnp.maximum(q, 0.0)
    # Safe norm: avoids NaN gradient of ||x||/||x|| at x=0 (happens when pusher
    # lands exactly on a box corner).  1e-18 is negligible vs contact tolerances.
    outside = jnp.sqrt(jnp.dot(outside_q, outside_q) + 1e-16)
    inside = jnp.minimum(jnp.maximum(q[0], q[1]), 0.0)
    return outside + inside


def _t_sdf(p_body: jnp.ndarray) -> jnp.ndarray:
    """SDF of the T block (union of top bar + stem) in the block body frame."""
    sdf_top = _box_sdf_2d(p_body, T_TOP_BAR_CENTER, T_TOP_BAR_HALF)
    sdf_stem = _box_sdf_2d(p_body, T_STEM_CENTER, T_STEM_HALF)
    return jnp.minimum(sdf_top, sdf_stem)


_t_sdf_value_and_grad = jax.value_and_grad(_t_sdf)


def _pusher_t_contact_force_single(
    pusher_xy: jnp.ndarray,
    pusher_vel_xy: jnp.ndarray,
    t_xy: jnp.ndarray,
    t_vel_xy: jnp.ndarray,
    t_theta: jnp.ndarray,
    t_omega: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Penalty spring-damper contact between the pusher and the T block.

    Returns
    -------
    F_pusher_world : (2,) force on the pusher in world xy (N)
    F_t_world      : (2,) force on the T-block in world xy (N)
    tau_t          : scalar torque on the T-block about +z at the block's joint axis (N·m)
    """
    rel_world = pusher_xy - t_xy
    c = jnp.cos(t_theta)
    s = jnp.sin(t_theta)
    R_world_to_body = jnp.array([[c, s], [-s, c]])
    R_body_to_world = jnp.array([[c, -s], [s, c]])
    p_body = R_world_to_body @ rel_world

    sdf, grad_body = _t_sdf_value_and_grad(p_body)
    n_body = grad_body / (jnp.linalg.norm(grad_body) + 1e-12)
    n_world = R_body_to_world @ n_body

    penetration = jnp.maximum(0.0, PUSHER_EFFECTIVE_RADIUS - sdf)

    # Velocity of the material point of the T that's momentarily at the pusher location.
    v_t_at_contact = t_vel_xy + t_omega * jnp.array([-rel_world[1], rel_world[0]])
    v_rel = pusher_vel_xy - v_t_at_contact
    closing_rate = -jnp.dot(v_rel, n_world)

    # Penalty spring + damping. Damping only applies while penetrating and only
    # resists closing (never yanks bodies together).
    is_contact = penetration > 0.0
    f_mag = jnp.where(
        is_contact,
        T_PUSHER_CONTACT_K * penetration + T_PUSHER_CONTACT_D * jnp.maximum(closing_rate, 0.0),
        0.0,
    )

    F_pusher_world = f_mag * n_world
    F_t_world = -F_pusher_world
    # Torque about +z from applying F_t at the contact point (approximated by the
    # pusher location relative to the T-joint axis).
    tau_t = rel_world[0] * F_t_world[1] - rel_world[1] * F_t_world[0]

    return F_pusher_world, F_t_world, tau_t


_pusher_t_contact_force_batched = jax.jit(jax.vmap(_pusher_t_contact_force_single, in_axes=(0, 0, 0, 0, 0, 0)))


def _t_floor_friction_single(
    t_vel_xy: jnp.ndarray,
    t_omega: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Coulomb + viscous friction between the T-block and the floor.

    Returns (F_fric_world (2,), tau_fric). Forces act on the T-block's
    `t_block_joint_x/y/theta` joints (which are world-axis aligned).
    """
    coulomb_lin = T_FLOOR_MU * T_MASS * _GRAVITY_MAG
    coulomb_rot = T_FLOOR_MU * T_MASS * _GRAVITY_MAG * T_FLOOR_ROT_RADIUS

    F_fric = -coulomb_lin * jnp.tanh(t_vel_xy / T_FLOOR_V_EPS) - T_FLOOR_C_LIN * t_vel_xy
    tau_fric = -coulomb_rot * jnp.tanh(t_omega / T_FLOOR_W_EPS) - T_FLOOR_C_ROT * t_omega
    return F_fric, tau_fric


_t_floor_friction_batched = jax.jit(jax.vmap(_t_floor_friction_single, in_axes=(0, 0)))


def _t_wall_contact_forces_single(
    t_xy: jnp.ndarray,
    t_vel_xy: jnp.ndarray,
    t_theta: jnp.ndarray,
    t_omega: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Spring-damper penalty forces from T corners vs the 4 workspace walls.

    Returns F_wall_world (2,), tau_wall.
    """
    c = jnp.cos(t_theta)
    s = jnp.sin(t_theta)
    R = jnp.array([[c, -s], [s, c]])  # body -> world
    corners_world = t_xy[None, :] + (_T_CORNERS_JAX @ R.T)  # (8, 2)

    r = corners_world - t_xy[None, :]  # (8, 2) moment arms in world frame
    v_corners = t_vel_xy[None, :] + t_omega * jnp.stack([-r[:, 1], r[:, 0]], axis=-1)  # (8, 2)

    # Penetration depths into each wall (zero when not in contact).
    pen_left = jnp.maximum(0.0, -corners_world[:, 0])  # x < 0
    pen_right = jnp.maximum(0.0, corners_world[:, 0] - WORKSPACE_WIDTH)  # x > W
    pen_bottom = jnp.maximum(0.0, -corners_world[:, 1])  # y < 0
    pen_top = jnp.maximum(0.0, corners_world[:, 1] - WORKSPACE_HEIGHT)  # y > H

    # Closing rates (positive = moving toward the wall).
    cr_left = jnp.maximum(0.0, -v_corners[:, 0])
    cr_right = jnp.maximum(0.0, v_corners[:, 0])
    cr_bottom = jnp.maximum(0.0, -v_corners[:, 1])
    cr_top = jnp.maximum(0.0, v_corners[:, 1])

    # Force magnitudes; damping only applies while in contact.
    f_left = jnp.where(pen_left > 0.0, T_WALL_CONTACT_K * pen_left + T_WALL_CONTACT_D * cr_left, 0.0)
    f_right = jnp.where(pen_right > 0.0, T_WALL_CONTACT_K * pen_right + T_WALL_CONTACT_D * cr_right, 0.0)
    f_bottom = jnp.where(pen_bottom > 0.0, T_WALL_CONTACT_K * pen_bottom + T_WALL_CONTACT_D * cr_bottom, 0.0)
    f_top = jnp.where(pen_top > 0.0, T_WALL_CONTACT_K * pen_top + T_WALL_CONTACT_D * cr_top, 0.0)

    # Net force per corner: left/bottom push +, right/top push -.
    F_corners = jnp.stack([f_left - f_right, f_bottom - f_top], axis=-1)  # (8, 2)

    # Torque per corner: r × F (2D cross product = rx*Fy - ry*Fx).
    tau_corners = r[:, 0] * F_corners[:, 1] - r[:, 1] * F_corners[:, 0]  # (8,)

    return F_corners.sum(axis=0), tau_corners.sum()


_t_wall_contact_forces_batched = jax.jit(jax.vmap(_t_wall_contact_forces_single, in_axes=(0, 0, 0, 0)))


# ── Action dataclass ──────────────────────────────────────────────────────────


_ArrayLike = "np.ndarray | jax.Array"


@dataclass
class Action:
    """
      Batched push action for nenvs parallel environments.

      Faces
      - A: 0, B: 1, C: 2, D: 3, E: 4, F: 5

       _____ C ______
    B |              | D
      |___       ____|
          |     |
          |     |
        A |     | E
          |_ F _|

      Fields (all shape (nenvs, 1)):
        face           int32  in {0, 1, 2, 3, 4, 5}
        contact_point  float32 in CONTACT_POINT_BOUNDS (parametrizes position along the chosen face edge)
        angle          float32 in ANGLE_BOUNDS (push direction in the face frame)
        push_distance  float32 in [0, 0.1] (metres)

    Arrays may be either `np.ndarray` or `jax.Array`.  The validation runs only
    under concrete (non-traced) values so that `Action` objects can still be
    used to carry JAX tracers (e.g. for `jax.grad` or `jax.jit`).
    """

    face: Any
    contact_point: Any
    angle: Any
    # push_distance:  Any

    @property
    def nenvs(self) -> int:
        return self.face.shape[0]

    def __post_init__(self):
        expected_shape = (self.nenvs, 1)
        for name, arr in (
            ("face", self.face),
            ("contact_point", self.contact_point),
            ("angle", self.angle),
        ):
            assert isinstance(arr, (np.ndarray, jax.Array)), f"{name} must be np.ndarray or jax.Array, got {type(arr)}"
            assert arr.shape == expected_shape, f"{name} must be {expected_shape}, got shape {arr.shape}"

        # Only do value-based sanity checks on concrete arrays; skip under a JAX
        # trace (`jax.core.Tracer`), where the values aren't available yet.
        def _is_concrete(x: Any) -> bool:
            return isinstance(x, np.ndarray) or (isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer))

        if _is_concrete(self.face):
            f = np.asarray(self.face)
            assert f.dtype == np.int32, f"face must be int32, got {f.dtype}"
            assert np.all((f >= 0) & (f <= 5)), f"face must be in (0, 1, 2, 3, 4, 5), got {f}"
        if _is_concrete(self.contact_point):
            cp = np.asarray(self.contact_point)
            assert cp.dtype == np.float32, f"contact_point must be float32, got {cp.dtype}"
            lo, hi = CONTACT_POINT_BOUNDS
            assert np.all((cp >= lo) & (cp <= hi)), f"contact_point must be in [{lo}, {hi}], got {cp}"
        if _is_concrete(self.angle):
            a = np.asarray(self.angle)
            assert a.dtype == np.float32, f"angle must be float32, got {a.dtype}"
            lo, hi = float(ANGLE_BOUNDS[0]), float(ANGLE_BOUNDS[1])
            assert np.all((a >= lo) & (a <= hi)), f"angle must be in [{lo}, {hi}], got {a}"


@dataclass
class ActionResult:
    """
    Result of applying an action to an environment.
    """

    action: Action
    t_poses: np.ndarray
    t_distances: np.ndarray

    @property
    def n_timesteps(self) -> int:
        return self.t_poses.shape[1]

    def __post_init__(self):
        assert isinstance(self.action, Action), "action must be an Action"
        assert self.t_poses.shape == (self.action.nenvs, self.n_timesteps, 3), (
            f"t_poses must be (nenvs, n_timesteps, 3), got shape {self.t_poses.shape}"
        )
        assert self.t_distances.shape == (self.action.nenvs, self.n_timesteps), (
            f"t_distances must be (nenvs, n_timesteps), got shape {self.t_distances.shape}"
        )


# ── Environment ───────────────────────────────────────────────────────────────


@jax.jit
def step_single(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    joint_force_references: jnp.ndarray,
) -> tuple[js.data.JaxSimModelData, dict[str, Any]]:
    return js.model.step(
        model=model,
        data=data,
        link_forces=None,
        joint_force_references=joint_force_references,
    )


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0, 0))
def step_parallel(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    joint_force_references: jnp.ndarray,
) -> tuple[js.data.JaxSimModelData, dict[str, Any]]:
    return step_single(
        model=model,
        data=data,
        joint_force_references=joint_force_references,
    )


def _compute_phase_steps(dt: float, n_sim_steps: int) -> tuple[int, int]:
    """Split `n_sim_steps` between approach / push phases.

    Returned values are concrete Python ints so they can be used as static
    shapes inside a jit'd function (needed for `jnp.linspace(..., num=N)`).
    """
    approach_full = max(1, int(round(PUSHER_APPROACH_DISTANCE / (PUSHER_APPROACH_VELOCITY * dt))))
    push_full = max(1, int(round(DEFAULT_PUSH_DISTANCE / (PUSHER_PUSH_VELOCITY * dt))))
    approach_steps = min(approach_full, n_sim_steps)
    push_steps = min(push_full, n_sim_steps - approach_steps)
    return approach_steps, push_steps


@functools.partial(
    jax.jit,
    static_argnames=(
        "approach_steps",
        "push_steps",
        "pusher_x_idx",
        "pusher_y_idx",
        "T_x_idx",
        "T_y_idx",
        "T_theta_idx",
        "T_target_x_idx",
        "T_target_y_idx",
        "T_target_theta_idx",
        "nenvs",
        "check_t_displacement",
    ),
)
def _step_pure_impl(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
    face: jnp.ndarray,  # (nenvs,) int
    contact_point: jnp.ndarray,  # (nenvs,)  must satisfy CONTACT_POINT_BOUNDS (checked below)
    angle: jnp.ndarray,  # (nenvs,)  must satisfy ANGLE_BOUNDS (checked below)
    static_joint_mask: jnp.ndarray,  # (dofs,) bool
    pusher_x_idx: int,
    pusher_y_idx: int,
    T_x_idx: int,
    T_y_idx: int,
    T_theta_idx: int,
    T_target_x_idx: int,
    T_target_y_idx: int,
    T_target_theta_idx: int,
    nenvs: int,
    approach_steps: int,
    push_steps: int,
    check_t_displacement: bool = True,
) -> tuple[js.data.JaxSimModelData, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fully pure, jittable, differentiable physics rollout for one action.

    Returns
    -------
    data_new : `JaxSimModelData` at the end of the rollout.
    t_poses  : (nenvs, n_sim_steps, 3) the T pose at every sim step.
    t_dists  : (nenvs, n_sim_steps) Euclidean xy distance from T to the goal
               (``t_target`` joint xy from ``data`` after pusher placement).
    jpos_traj: (nenvs, n_sim_steps, dofs) full joint position trace — useful
               only for replay into the visualizer / recorder; do NOT touch
               this when computing gradients (it's large and mostly redundant).
    """
    dt = model.time_step

    # ── Planning (all jnp) ─────────────────────────────────────────────
    contact_point = jnp.asarray(contact_point, dtype=jnp.float32)
    angle = jnp.asarray(angle, dtype=jnp.float32)
    jax.debug.callback(_raise_unless_contact_angle_bounds, contact_point, angle)
    T_x = data.joint_positions[:, T_x_idx]
    T_y = data.joint_positions[:, T_y_idx]
    T_theta = data.joint_positions[:, T_theta_idx]
    t_poses_now = jnp.stack([T_x, T_y, T_theta], axis=-1)
    pusher_start, pre_contact, push_direction = _plan_push_jax(t_poses_now, face, contact_point, angle, nenvs)

    # ── Pusher teleport + velocity reset ────────────────────────────────
    jp = data.joint_positions.at[:, pusher_x_idx].set(pusher_start[:, 0])
    jp = jp.at[:, pusher_y_idx].set(pusher_start[:, 1])
    jv = data.joint_velocities.at[:, pusher_x_idx].set(0.0)
    jv = jv.at[:, pusher_y_idx].set(0.0)
    data = data.replace(model=model, joint_positions=jp, joint_velocities=jv)

    # Pin reference + goal xy from this rollout's initial state (not from
    # ``env`` attributes) so :func:`jax.jit` costs see them as tracers when
    # ``data`` changes after :meth:`PushTEnv.reset`.
    pin_jp = data.joint_positions
    pin_bp = data.base_position
    pin_bq = data.base_quaternion
    pin_blv = data._base_linear_velocity
    pin_bav = data._base_angular_velocity
    target_xy = jnp.stack([pin_jp[:, T_target_x_idx], pin_jp[:, T_target_y_idx]], axis=-1)
    target_theta = pin_jp[:, T_target_theta_idx]
    cos_tgt = jnp.cos(target_theta)
    sin_tgt = jnp.sin(target_theta)
    R_tgt = jnp.stack([jnp.stack([cos_tgt, -sin_tgt], axis=-1), jnp.stack([sin_tgt, cos_tgt], axis=-1)], axis=1)
    target_corners = target_xy[:, None, :] + jnp.einsum("nij,kj->nki", R_tgt, _DIST_CORNERS_JAX)  # (nenvs, 4, 2)

    # ── Build pusher target trajectory ─────────────────────────────────
    approach_scales = jnp.linspace(0.0, 1.0, num=approach_steps)
    push_scales = jnp.linspace(0.0, DEFAULT_PUSH_DISTANCE, num=push_steps)

    approach_targets = (
        pusher_start[:, None, :] + approach_scales[None, :, None] * (pre_contact - pusher_start)[:, None, :]
    )
    push_targets = pre_contact[:, None, :] + push_scales[None, :, None] * push_direction[:, None, :]
    pusher_targets = jnp.concatenate([approach_targets, push_targets], axis=1)

    workspace_margin = PUSHER_RADIUS
    lb = jnp.array([workspace_margin, workspace_margin])
    ub = jnp.array([WORKSPACE_WIDTH - workspace_margin, WORKSPACE_HEIGHT - workspace_margin])
    pusher_targets = jnp.clip(pusher_targets, lb, ub)

    pusher_target_velocities = jnp.zeros_like(pusher_targets)
    pusher_target_velocities = pusher_target_velocities.at[:, 1:, :].set(
        (pusher_targets[:, 1:, :] - pusher_targets[:, :-1, :]) / dt
    )

    # ── Scan body: one sim step ─────────────────────────────────────────
    def sim_step(data: js.data.JaxSimModelData, step_input):
        target_xy_step, target_vel_xy = step_input  # each (nenvs, 2)
        cur_xy = jnp.stack(
            [data.joint_positions[:, pusher_x_idx], data.joint_positions[:, pusher_y_idx]],
            axis=-1,
        )
        cur_vel_xy = jnp.stack(
            [data.joint_velocities[:, pusher_x_idx], data.joint_velocities[:, pusher_y_idx]],
            axis=-1,
        )
        pusher_forces = PUSHER_KP * (target_xy_step - cur_xy) + PUSHER_KD * (target_vel_xy - cur_vel_xy)
        pusher_forces = jnp.clip(pusher_forces, -PUSHER_MAX_FORCE, PUSHER_MAX_FORCE)
        t_xy = jnp.stack([data.joint_positions[:, T_x_idx], data.joint_positions[:, T_y_idx]], axis=-1)
        t_vel_xy = jnp.stack(
            [data.joint_velocities[:, T_x_idx], data.joint_velocities[:, T_y_idx]],
            axis=-1,
        )
        t_theta = data.joint_positions[:, T_theta_idx]
        t_omega = data.joint_velocities[:, T_theta_idx]

        F_pusher_world, F_t_world, tau_t = jax.vmap(_pusher_t_contact_force_single)(
            cur_xy, cur_vel_xy, t_xy, t_vel_xy, t_theta, t_omega
        )
        F_fric_world, tau_fric = jax.vmap(_t_floor_friction_single)(t_vel_xy, t_omega)
        F_wall_world, tau_wall = jax.vmap(_t_wall_contact_forces_single)(t_xy, t_vel_xy, t_theta, t_omega)

        joint_forces = jnp.zeros_like(data.joint_velocities)
        joint_forces = joint_forces.at[:, pusher_x_idx].set(pusher_forces[:, 0] + F_pusher_world[:, 0])
        joint_forces = joint_forces.at[:, pusher_y_idx].set(pusher_forces[:, 1] + F_pusher_world[:, 1])
        joint_forces = joint_forces.at[:, T_x_idx].set(F_t_world[:, 0] + F_fric_world[:, 0] + F_wall_world[:, 0])
        joint_forces = joint_forces.at[:, T_y_idx].set(F_t_world[:, 1] + F_fric_world[:, 1] + F_wall_world[:, 1])
        joint_forces = joint_forces.at[:, T_theta_idx].set(tau_t + tau_fric + tau_wall)

        data = step_parallel(model, data, joint_forces)
        mask = static_joint_mask[None, :]
        jp_pinned = jnp.where(mask, pin_jp, data.joint_positions)
        jv_pinned = jnp.where(mask, 0.0, data.joint_velocities)
        data = data.replace(
            model=model,
            base_position=pin_bp,
            base_quaternion=pin_bq,
            base_linear_velocity=pin_blv,
            base_angular_velocity=pin_bav,
            joint_positions=jp_pinned,
            joint_velocities=jv_pinned,
        )

        tx = data.joint_positions[:, T_x_idx]
        ty = data.joint_positions[:, T_y_idx]
        tth = data.joint_positions[:, T_theta_idx]
        t_pose = jnp.stack([tx, ty, tth], axis=-1)
        cos_t = jnp.cos(tth)
        sin_t = jnp.sin(tth)
        R_t = jnp.stack([jnp.stack([cos_t, -sin_t], axis=-1), jnp.stack([sin_t, cos_t], axis=-1)], axis=1)
        t_corners = jnp.stack([tx, ty], axis=-1)[:, None, :] + jnp.einsum("nij,kj->nki", R_t, _DIST_CORNERS_JAX)
        t_dist = jnp.sum((t_corners - target_corners) ** 2, axis=(-1, -2))  # (nenvs,)

        return data, (t_pose, t_dist, data.joint_positions)

    # Scan drives the time dimension first.
    scan_in = (
        jnp.swapaxes(pusher_targets, 0, 1),  # (n_sim_steps, nenvs, 2)
        jnp.swapaxes(pusher_target_velocities, 0, 1),  # (n_sim_steps, nenvs, 2)
    )
    data_final, (t_poses_T, t_dists_T, jpos_T) = jax.lax.scan(sim_step, data, scan_in)

    t_poses = jnp.swapaxes(t_poses_T, 0, 1)  # (nenvs, n_sim_steps, 3)
    t_dists = jnp.swapaxes(t_dists_T, 0, 1)  # (nenvs, n_sim_steps)
    jpos_traj = jnp.swapaxes(jpos_T, 0, 1)  # (nenvs, n_sim_steps, dofs)

    # Optional sanity check: successful pushes should move the T block in xy.
    if check_t_displacement:
        t_displacement = jnp.linalg.norm(t_poses[:, -1, :2] - t_poses_now[:, :2], axis=-1)

        def _assert_t_moved(disp: np.ndarray) -> None:
            nan_envs = np.where(np.isnan(disp))[0]
            if len(nan_envs):
                print(f"WARNING: NaN displacement in envs {nan_envs} — physics diverged")
            no_move = np.where(~np.isnan(disp) & (disp <= 1e-3))[0]
            assert len(no_move) == 0, f"T block did not move in envs {no_move} (displacements: {disp})"

        jax.debug.callback(_assert_t_moved, t_displacement)

    return data_final, t_poses, t_dists, jpos_traj


class PushTEnv:
    """
    Parallel Push-T environment.

    State: (nenvs, 3) array of [x, y, theta] poses.
    - use_relative_coordinates: if True, the pose and velocity of the T in the context vector is given in the target T's
    frame, not the global frame.
    """

    def __init__(self, nenvs: int, use_relative_coordinates: bool = False, record_video: bool = False, visualize: bool = False):
        self._nenvs = nenvs
        self._use_relative_coordinates = use_relative_coordinates
        self._record_video = record_video
        self._visualize = visualize
        self._visualizer = None
        self._viewer = None
        self._t_poses: np.ndarray = np.zeros((nenvs, 3), dtype=np.float32)
        self._target_poses: np.ndarray = np.zeros((nenvs, 3), dtype=np.float32)
        self._frames: list = []

        rod_sdf = rod.Sdf.load(Path("assets/scene.sdf"))
        for model in rod_sdf.models():
            model.switch_frame_convention(frame_convention=rod.FrameConvention.Urdf, explicit_frames=True)
        model_sdf_string = rod_sdf.serialize(pretty=True)

        self._model = js.model.JaxSimModel.build_from_model_description(
            model_description=model_sdf_string,
            time_step=0.001,
            contact_params=_jsc.SoftContactsParams.build(mu=0.0),
        )

        joint_names = self._model.joint_names()
        self._T_x_idx = joint_names.index("t_block_joint_x")
        self._T_y_idx = joint_names.index("t_block_joint_y")
        self._T_theta_idx = joint_names.index("t_block_joint_theta")
        self._T_target_x_idx = joint_names.index("t_target_block_joint_x")
        self._T_target_y_idx = joint_names.index("t_target_block_joint_y")
        self._T_target_theta_idx = joint_names.index("t_target_block_joint_theta")
        self._pusher_x_idx = joint_names.index("pusher_joint_x")
        self._pusher_y_idx = joint_names.index("pusher_joint_y")

        # "Active" joints are the ones whose state is driven by our physics
        # (pusher PD control + T block custom contact/friction).  Everything
        # else (walls, target-T) has no physical driver and is only held in
        # place by soft joint limits that JaxSim does not enforce rigidly —
        # so they accumulate integrator drift over long rollouts.  We pin
        # them back to their reset values every sim step.
        active_joint_idxs = {
            self._T_x_idx,
            self._T_y_idx,
            self._T_theta_idx,
            self._pusher_x_idx,
            self._pusher_y_idx,
        }
        dofs = self._model.dofs()
        static_mask_np = np.array([i not in active_joint_idxs for i in range(dofs)], dtype=bool)
        self._static_joint_mask = jnp.asarray(static_mask_np)  # (dofs,)

        key = jax.random.PRNGKey(seed=0)
        key, *subkeys = jax.random.split(key=key, num=nenvs + 1)
        data_batch_t0 = jax.vmap(
            lambda k: js.data.random_model_data(
                model=self._model,
                key=k,
                base_pos_bounds=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
                base_vel_lin_bounds=(0, 0),
                base_vel_ang_bounds=(0, 0),
            )
        )(jnp.vstack(subkeys))

        row_length = int(jnp.sqrt(nenvs))
        spacing = max(WORKSPACE_WIDTH, WORKSPACE_HEIGHT) * 1.2
        row_dist = spacing * (row_length - 1) / 2
        x, y = jnp.meshgrid(
            jnp.linspace(-row_dist, row_dist, num=row_length) if row_length > 1 else jnp.array([0.0]),
            jnp.linspace(-row_dist, row_dist, num=row_length) if row_length > 1 else jnp.array([0.0]),
        )
        xy_coordinate = jnp.stack([x.flatten(), y.flatten()], axis=-1)
        self._xy_centers = np.array(xy_coordinate)
        self._data = data_batch_t0.replace(
            model=self._model,
            base_position=data_batch_t0.base_position.at[:, :2].set(xy_coordinate),
        )

        self.reset()

        if self._visualize or record_video:
            _grid_span = spacing * (row_length - 1) + max(WORKSPACE_WIDTH, WORKSPACE_HEIGHT)
            _cam_kwargs = dict(
                lookat=[WORKSPACE_WIDTH / 2, WORKSPACE_HEIGHT / 2, 0.0],
                distance=float(_grid_span / (2 * np.tan(np.deg2rad(22.5))) * 1.1),
                azimuth=90,
                elevation=-90,
            )

            convert_kwargs = {}
            if record_video:
                convert_kwargs["cameras"] = jaxsim.mujoco.loaders.MujocoCamera.build_from_target_view(
                    camera_name="t_block_cam", **_cam_kwargs
                )
            mjcf_string, assets = jaxsim.mujoco.ModelToMjcf.convert(self._model.built_from, **convert_kwargs)

            parser = etree.XMLParser(remove_blank_text=True)
            root = etree.fromstring(text=mjcf_string.encode(), parser=parser)
            worldbody = root.find("worldbody")
            env_center_original = worldbody.find("body[@name='env_center']")
            worldbody.remove(env_center_original)
            for i in range(nenvs):
                new_body = copy.deepcopy(env_center_original)
                for el in new_body.iter():
                    if "name" in el.attrib:
                        el.set("name", f"{el.get('name')}_{i}")
                worldbody.append(new_body)
            actuator = root.find("actuator")
            if actuator is not None:
                root.remove(actuator)
            for light in root.findall(".//light"):
                if light.get("target") == "env_center":
                    light.set("target", "env_center_0")
            for camera in root.findall(".//camera"):
                if camera.get("target") == "env_center":
                    camera.set("target", "env_center_0")
            multi_xml = etree.tostring(root, pretty_print=True).decode()
            self._mj_multi_model = mj.MjModel.from_xml_string(multi_xml, assets=assets)
            self._mj_multi_model.opt.gravity[:] = 0.0  # Prevent bodies drifting in MuJoCo physics step
            self._mj_multi_data = mj.MjData(self._mj_multi_model)

        if self._visualize:
            self._visualizer = jaxsim.mujoco.MujocoVisualizer(model=self._mj_multi_model, data=self._mj_multi_data)
            self._viewer = self._visualizer.open_viewer()
            jaxsim.mujoco.MujocoVisualizer.setup_viewer_camera(self._viewer, **_cam_kwargs)
            self._sync_visualizer()
        else:
            self._visualizer = None
            self._viewer = None

        if record_video:
            self._recorder = jaxsim.mujoco.MujocoVideoRecorder(
                model=self._mj_multi_model,
                data=self._mj_multi_data,
                fps=int(1 / self._model.time_step / 12),
                width=2512,
                height=2512
            )

    @property
    def nenvs(self) -> int:
        return self._nenvs

    @property
    def t_poses(self) -> np.ndarray:
        """Current poses (nenvs, 3) — read-only copy."""
        return self._t_poses.copy()

    @property
    def target_poses(self) -> np.ndarray:
        """Target poses (nenvs, 3) — read-only copy."""
        return self._target_poses.copy()

    @property
    def t_target_poses(self) -> np.ndarray:
        """Target pose (3) — read-only copy."""
        return self._target_poses.copy()

    @property
    def data(self) -> js.data.JaxSimModelData:
        """Current `JaxSimModelData` — pass this to :py:meth:`step_pure` when
        you want to differentiate through a rollout starting from the current
        simulation state.
        """
        return self._data

    @property
    def model(self) -> js.model.JaxSimModel:
        return self._model

    def get_context_vector(self, data) -> jnp.ndarray:

        if self._use_relative_coordinates:
            target_x = data.joint_positions[:, self._T_target_x_idx]
            target_y = data.joint_positions[:, self._T_target_y_idx]
            target_theta = data.joint_positions[:, self._T_target_theta_idx]

            t_x = data.joint_positions[:, self._T_x_idx]
            t_y = data.joint_positions[:, self._T_y_idx]
            t_theta = data.joint_positions[:, self._T_theta_idx]

            vx = data.joint_velocities[:, self._T_x_idx]
            vy = data.joint_velocities[:, self._T_y_idx]
            vtheta = data.joint_velocities[:, self._T_theta_idx]

            dx = t_x - target_x
            dy = t_y - target_y
            cos_t = jnp.cos(target_theta)
            sin_t = jnp.sin(target_theta)

            rel_x = dx * cos_t + dy * sin_t
            rel_y = -dx * sin_t + dy * cos_t
            rel_theta = t_theta - target_theta
            rel_vx = vx * cos_t + vy * sin_t
            rel_vy = -vx * sin_t + vy * cos_t

            return jnp.stack(
                [target_x, target_y, target_theta, rel_x, rel_y, rel_theta, rel_vx, rel_vy, vtheta],
                axis=-1,
            )

        else:
            return jnp.concatenate(
                [
                    jnp.stack(
                        [
                            data.joint_positions[:, self._T_target_x_idx],
                            data.joint_positions[:, self._T_target_y_idx],
                            data.joint_positions[:, self._T_target_theta_idx],
                        ],
                        axis=-1,
                    ),  # target pose
                    jnp.stack(
                        [
                            data.joint_positions[:, self._T_x_idx],
                            data.joint_positions[:, self._T_y_idx],
                            data.joint_positions[:, self._T_theta_idx],
                        ],
                        axis=-1,
                    ),  # T pose
                    jnp.stack(
                        [
                            data.joint_velocities[:, self._T_x_idx],
                            data.joint_velocities[:, self._T_y_idx],
                            data.joint_velocities[:, self._T_theta_idx],
                        ],
                        axis=-1,
                    ),  # T velocity
                    # jnp.stack(
                    #     [data.joint_positions[:, self._pusher_x_idx], data.joint_positions[:, self._pusher_y_idx]], axis=-1
                    # ),  # pusher xy
                ],
                axis=-1,
            )

    def _sync_visualizer(self) -> None:
        if self._visualizer is None:
            return
        self._sync_visualizer_from_positions(
            base_position=np.asarray(self._data.base_position),
            base_orientation=np.asarray(self._data.base_orientation),
            joint_positions=np.asarray(self._data.joint_positions),
        )

    def _update_multi_env_data(
        self,
        base_position: np.ndarray,  # (nenvs, 3)
        base_orientation: np.ndarray,  # (nenvs, 4)
        joint_positions: np.ndarray,  # (nenvs, dofs)
    ) -> None:
        joint_names = self._model.joint_names()
        for i in range(self._nenvs):
            jnt_id = mj.mj_name2id(self._mj_multi_model, mj.mjtObj.mjOBJ_JOINT, f"world_to_base_{i}")
            if jnt_id != -1:
                qpos_adr = self._mj_multi_model.jnt_qposadr[jnt_id]
                self._mj_multi_data.qpos[qpos_adr : qpos_adr + 3] = base_position[i]
                self._mj_multi_data.qpos[qpos_adr + 3 : qpos_adr + 7] = base_orientation[i]
            for orig_jnt_idx, jnt_name in enumerate(joint_names):
                multi_jnt_id = mj.mj_name2id(self._mj_multi_model, mj.mjtObj.mjOBJ_JOINT, f"{jnt_name}_{i}")
                if multi_jnt_id != -1:
                    qpos_adr = self._mj_multi_model.jnt_qposadr[multi_jnt_id]
                    self._mj_multi_data.qpos[qpos_adr] = joint_positions[i, orig_jnt_idx]

    def _sync_visualizer_from_positions(
        self,
        base_position: np.ndarray,
        base_orientation: np.ndarray,
        joint_positions: np.ndarray,
    ) -> None:
        if self._visualizer is None:
            return
        self._update_multi_env_data(base_position, base_orientation, joint_positions)
        self._visualizer.sync(self._viewer)

    def reset(
        self, seed: int | None = None, target_poses: np.ndarray | None = None, t_poses: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Reset all environments to random orientations at the origin.
        Returns poses (nenvs, 3).
        """
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        rng = np.random.default_rng(seed)

        # First, set randomized poses for the T and target T
        if target_poses is None:
            self._target_poses = np.zeros((self._nenvs, 3), dtype=np.float32)
            self._target_poses[:, 0] = rng.uniform(T_RADIUS, WORKSPACE_WIDTH - T_RADIUS, size=self._nenvs)
            self._target_poses[:, 1] = rng.uniform(T_RADIUS, WORKSPACE_HEIGHT - T_RADIUS, size=self._nenvs)
            self._target_poses[:, 2] = rng.uniform(-np.pi, np.pi, size=self._nenvs)
        else:
            assert target_poses.shape == (self._nenvs, 3), (
                f"target_poses must be ({self._nenvs}, 3), got {target_poses.shape}"
            )
            self._target_poses = target_poses

        #
        if t_poses is None:
            self._t_poses = np.zeros((self._nenvs, 3), dtype=np.float32)
            self._t_poses[:, 0] = rng.uniform(T_RADIUS, WORKSPACE_WIDTH - T_RADIUS, size=self._nenvs)
            self._t_poses[:, 1] = rng.uniform(T_RADIUS, WORKSPACE_HEIGHT - T_RADIUS, size=self._nenvs)
            self._t_poses[:, 2] = rng.uniform(-np.pi, np.pi, size=self._nenvs)
        else:
            assert t_poses.shape == (self._nenvs, 3), f"t_poses must be ({self._nenvs}, 3), got {t_poses.shape}"
            self._t_poses = t_poses

        # Reset video frames
        self._frames = []

        # The base is env_center at xy_centers, with identity orientation
        new_base_position = self._data.base_position.at[:, :2].set(self._xy_centers)
        new_base_position = new_base_position.at[:, 2].set(0.01)

        # Identity quaternions for env_center
        quats = jnp.zeros_like(self._data.base_orientation)
        quats = quats.at[:, 0].set(1.0)

        # Update joint positions for t_block and pusher
        new_joint_positions = jnp.zeros_like(self._data.joint_positions)
        new_joint_positions = new_joint_positions.at[:, self._T_x_idx].set(self._t_poses[:, 0])
        new_joint_positions = new_joint_positions.at[:, self._T_y_idx].set(self._t_poses[:, 1])
        new_joint_positions = new_joint_positions.at[:, self._T_theta_idx].set(self._t_poses[:, 2])
        new_joint_positions = new_joint_positions.at[:, self._T_target_x_idx].set(self._target_poses[:, 0])
        new_joint_positions = new_joint_positions.at[:, self._T_target_y_idx].set(self._target_poses[:, 1])
        new_joint_positions = new_joint_positions.at[:, self._T_target_theta_idx].set(self._target_poses[:, 2])
        new_joint_positions = new_joint_positions.at[:, self._pusher_x_idx].set(0.75)
        new_joint_positions = new_joint_positions.at[:, self._pusher_y_idx].set(0.75)

        # Reset internal JaxSim data
        self._data = self._data.replace(
            model=self._model,
            base_position=new_base_position,
            base_quaternion=quats,
            joint_positions=new_joint_positions,
            base_linear_velocity=jnp.zeros_like(self._data._base_linear_velocity),
            base_angular_velocity=jnp.zeros_like(self._data._base_angular_velocity),
            joint_velocities=jnp.zeros_like(self._data.joint_velocities),
        )
        # Cache the pinned base state so every sim step can re-clamp it.
        # `env_center` is structurally a floating base in the SDF, so any joint
        # force applied on descendant joints (pusher, T) creates a reaction that
        # drifts env_center. We pin it every sim step to behave as a fixed base.
        self._pinned_base_position = self._data.base_position
        self._pinned_base_quaternion = self._data.base_quaternion
        self._pinned_base_linear_velocity = jnp.zeros_like(self._data._base_linear_velocity)
        self._pinned_base_angular_velocity = jnp.zeros_like(self._data._base_angular_velocity)
        # Cache pinned joint positions for the "static" joints (walls,
        # target-T) so every sim step can clamp them back and prevent drift.
        self._pinned_joint_positions = self._data.joint_positions
        zero_forces = jnp.zeros((self._nenvs, self._model.dofs()))
        self._data = step_parallel(self._model, self._data, zero_forces)
        self._data = self._pin_static_assets(self._data)
        self._t_poses = np.column_stack(
            [
                np.asarray(self._data.joint_positions[:, self._T_x_idx]),
                np.asarray(self._data.joint_positions[:, self._T_y_idx]),
                np.asarray(self._data.joint_positions[:, self._T_theta_idx]),
            ]
        )
        self._sync_visualizer()
        return self._t_poses.copy()

    def _pin_static_assets(self, data: js.data.JaxSimModelData) -> js.data.JaxSimModelData:
        """Force env_center back to its fixed world pose with zero velocity
        and re-clamp the "static" joints (walls, target-T) to their reset
        values.  Needed because JaxSim's articulated dynamics let small
        numerical noise accumulate on joints whose soft limits are supposed
        to hold them fixed.
        """
        mask = self._static_joint_mask[None, :]  # (1, dofs)
        jp = jnp.where(mask, self._pinned_joint_positions, data.joint_positions)
        jv = jnp.where(mask, 0.0, data.joint_velocities)
        return data.replace(
            model=self._model,
            base_position=self._pinned_base_position,
            base_quaternion=self._pinned_base_quaternion,
            base_linear_velocity=self._pinned_base_linear_velocity,
            base_angular_velocity=self._pinned_base_angular_velocity,
            joint_positions=jp,
            joint_velocities=jv,
        )

    # ── Pure, differentiable step ────────────────────────────────────────
    #
    # The recommended entry-point for gradient-based optimization.
    #
    # Example — take the gradient of a final-distance cost w.r.t. the action:
    #
    #     env = PushTEnv(nenvs=N)
    #     env.reset()
    #     data0 = env.data                      # snapshot the initial JaxSim state
    #
    #     def cost(contact_point, angle, face):
    #         _, _, t_dists, _ = env.step_pure(
    #             data0, face, contact_point, angle, n_sim_steps=100
    #         )
    #         # Cost = distance-to-target at the final sim step, summed over envs
    #         return t_dists[:, -1].sum()
    #
    #     g_contact, g_angle = jax.grad(cost, argnums=(0, 1))(
    #         contact_point, angle, face
    #     )
    #
    # `face` is an integer index — `jax.grad` will return zeros for it.
    # `step_pure` is pure and has *no* side-effects: it does not touch
    # `self._data`, the visualizer, or the recorder.

    def step_pure(
        self,
        data: js.data.JaxSimModelData,
        face: jnp.ndarray,  # (nenvs,) int
        contact_point: jnp.ndarray,  # (nenvs,) float
        angle: jnp.ndarray,  # (nenvs,) float
        n_sim_steps: int = 100,
        *,
        check_t_displacement: bool = True,
    ):
        """Pure, jittable, differentiable single-action rollout.

        Returns ``(data_new, t_poses, t_distances, joint_positions_traj)`` —
        all JAX arrays.  Shapes:

        - ``t_poses``  : (nenvs, n_sim_steps, 3)
        - ``t_distances`` : (nenvs, n_sim_steps)
        - ``joint_positions_traj`` : (nenvs, n_sim_steps, dofs)

        Use this when you want to take gradients (e.g. ``jax.grad``) through
        a full 100-step rollout.  The ``face`` argument is an integer index
        and carries zero gradient; differentiate w.r.t. ``contact_point`` and
        ``angle`` (and optionally fields of ``data``).

        ``contact_point`` and ``angle`` must lie in :data:`CONTACT_POINT_BOUNDS` and
        :data:`ANGLE_BOUNDS` (same as :meth:`step` / :class:`Action`); otherwise the
        rollout raises ``ValueError``.

        Goal xy and static-joint pins are read from ``data`` inside the rollout
        (after pusher placement), so jitted costs stay correct across
        :meth:`reset` without threading extra ``env`` arrays.
        """
        assert n_sim_steps > 0, "n_sim_steps must be > 0"
        approach_steps, push_steps = _compute_phase_steps(float(self._model.time_step), n_sim_steps)
        return _step_pure_impl(
            self._model,
            data,
            jnp.asarray(face).astype(jnp.int32),
            jnp.asarray(contact_point).astype(jnp.float32),
            jnp.asarray(angle).astype(jnp.float32),
            self._static_joint_mask,
            self._pusher_x_idx,
            self._pusher_y_idx,
            self._T_x_idx,
            self._T_y_idx,
            self._T_theta_idx,
            self._T_target_x_idx,
            self._T_target_y_idx,
            self._T_target_theta_idx,
            self._nenvs,
            approach_steps,
            push_steps,
            check_t_displacement,
        )

    def step_pure_soft(
        self,
        data: js.data.JaxSimModelData,
        face_weights: jnp.ndarray,  # (nenvs, 6) float — per-face weights (e.g. softmax)
        contact_point: jnp.ndarray,  # (nenvs,) float
        angle: jnp.ndarray,  # (nenvs,) float
        n_sim_steps: int = 100,
        *,
        check_t_displacement: bool = True,
    ) -> tuple[js.data.JaxSimModelData, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Differentiable-face variant of :py:meth:`step_pure`.

        Same returns as `step_pure`, but `face_weights` is a `(nenvs, 6)` float
        array of per-face weights instead of an integer index. Internally,
        planning takes a convex combination of the 6 face start/end points
        (``weights @ face_points``), so gradients flow into `face_weights`.

        Typical usage — optimise over unconstrained `face_logits` and map to
        weights with `jax.nn.softmax(face_logits, axis=-1)`. Passing a one-hot
        matrix recovers the hard-face behaviour (up to float precision).

        ``contact_point`` and ``angle`` must lie in :data:`CONTACT_POINT_BOUNDS` and
        :data:`ANGLE_BOUNDS`; otherwise the rollout raises ``ValueError``.
        """
        assert n_sim_steps > 0, "n_sim_steps must be > 0"
        assert face_weights.shape == (self._nenvs, NUM_FACES), (
            f"face_weights must be ({self._nenvs}, NUM_FACES), got {face_weights.shape}"
        )

        approach_steps, push_steps = _compute_phase_steps(float(self._model.time_step), n_sim_steps)
        return _step_pure_impl(
            self._model,
            data,
            jnp.asarray(face_weights).astype(jnp.float32),
            jnp.asarray(contact_point).astype(jnp.float32),
            jnp.asarray(angle).astype(jnp.float32),
            self._static_joint_mask,
            self._pusher_x_idx,
            self._pusher_y_idx,
            self._T_x_idx,
            self._T_y_idx,
            self._T_theta_idx,
            self._T_target_x_idx,
            self._T_target_y_idx,
            self._T_target_theta_idx,
            self._nenvs,
            approach_steps,
            push_steps,
            check_t_displacement,
        )

    def step(self, action: Action, n_sim_steps: int = 100, *, check_t_displacement: bool = True) -> ActionResult:
        """
        Stateful wrapper around :py:meth:`step_pure`.  Advances the internal
        simulation state by one action, handles visualization and video
        recording (as post-hoc replays of the trajectory), and returns an
        ``ActionResult`` with NumPy arrays for convenience.

        For gradient-based use cases, prefer :py:meth:`step_pure`.
        """
        assert isinstance(action, Action), "action must be an Action"
        assert action.nenvs == self._nenvs, f"action.nenvs != self._nenvs ({action.nenvs} != {self._nenvs})"
        assert n_sim_steps > 0, "n_sim_steps must be > 0"

        # Extract (nenvs,)-shaped arrays from the Action, tolerating both
        # np.ndarray and jax.Array inputs.
        face = jnp.asarray(np.asarray(action.face)[:, 0]).astype(jnp.int32)
        contact_point = jnp.asarray(np.asarray(action.contact_point)[:, 0]).astype(jnp.float32)
        angle = jnp.asarray(np.asarray(action.angle)[:, 0]).astype(jnp.float32)

        self._data, t_poses, t_distances, jpos_traj = self.step_pure(
            data=self._data,
            face=face,
            contact_point=contact_point,
            angle=angle,
            n_sim_steps=n_sim_steps,
            check_t_displacement=check_t_displacement,
        )
        self._t_poses = np.asarray(t_poses[:, -1, :])

        # ── Post-hoc replay into viz / recorder ─────────────────────────
        if self._visualizer is not None or self._record_video:
            # Bring the trajectory to host once; cheap for the shapes we use.
            jpos_traj_np = np.asarray(jpos_traj)  # (nenvs, n_sim_steps, dofs)
            base_pos_np = np.asarray(self._data.base_position)
            base_quat_np = np.asarray(self._data.base_orientation)
            joint_names = self._model.joint_names()

            if self._record_video:
                for step_idx in range(n_sim_steps):
                    self._update_multi_env_data(base_pos_np, base_quat_np, jpos_traj_np[:, step_idx, :])
                    self._recorder.record_frame(camera_name="t_block_cam")

            if self._visualizer is not None:
                import time

                dt_sleep = float(self._model.time_step)
                for step_idx in range(n_sim_steps):
                    self._sync_visualizer_from_positions(base_pos_np, base_quat_np, jpos_traj_np[:, step_idx, :])
                    time.sleep(dt_sleep)

        return ActionResult(
            action=action,
            t_poses=np.asarray(t_poses),
            t_distances=np.asarray(t_distances),
        )

    def save_video(self, filename: str) -> None:
        """Write a video to filename."""
        assert self._record_video, "record_video must be True to save video"
        assert len(self._recorder.frames) > 0, "No frames to save"
        imageio.mimwrite(filename, self._recorder.frames, fps=self._recorder.fps)
        self._recorder.frames.clear()

    def save_video_from_jpos_traj(self, filename: str, jpos_traj: np.ndarray) -> None:
        """Write a video to filename from a given joint position trajectory."""
        assert self._record_video, "record_video must be True to save video"
        assert isinstance(filename, (str, Path)), f"filename must be a string or Path, got {type(filename)}"
        assert isinstance(jpos_traj, (np.ndarray, jax.Array)), (
            f"jpos_traj must be a numpy array or jax array, got {type(jpos_traj)}"
        )
        base_pos_np = np.asarray(self._data.base_position)
        base_quat_np = np.asarray(self._data.base_orientation)
        frames = []
        for step_idx in range(jpos_traj.shape[1]):
            self._update_multi_env_data(base_pos_np, base_quat_np, jpos_traj[:, step_idx, :])
            frame = self._recorder.render_frame(camera_name="t_block_cam")
            frames.append(frame)
        assert len(frames) > 0, "No frames to save"
        imageio.mimwrite(filename, frames, fps=self._recorder.fps)
