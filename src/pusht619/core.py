"""
Push-T parallel environment.

Block state: (x, y, theta) where (x, y) is the link-frame origin in world
coordinates and theta is the CCW rotation in radians.

Faces (A–F) go around the perimeter of the T; corners p0–p7 are at floor
level (z = 0) in the body frame defined in the URDF / README.
"""

from dataclasses import dataclass
import pathlib
import numpy as np
import functools
import os
import time
from typing import Any

import imageio
import jax
import jax.numpy as jnp
import jaxsim.api as js
import jaxsim.mujoco
import matplotlib.pyplot as plt
import mujoco as mj
import rod



# ── Action dataclass ──────────────────────────────────────────────────────────

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
      contact_point  float32 in [0, 1]   (0 = start corner, 1 = end corner)
      angle          float32 in [0, π]   (0 = tangent, π/2 = into block)
      push_distance  float32 in [0, 0.1] (metres)
    """
    face:           np.ndarray
    contact_point:  np.ndarray
    angle:          np.ndarray
    push_distance:  np.ndarray

    @property
    def nenvs(self) -> int:
        return len(self.face)

    def __post_init__(self):
        for name, arr in (
            ("face",          self.face),
            ("contact_point", self.contact_point),
            ("angle",         self.angle),
            ("push_distance", self.push_distance),
        ):
            assert isinstance(arr, np.ndarray), f"{name} must be np.ndarray"
            assert arr.shape == (self.nenvs, 1), f"{name} must be (nenvs, 1), got shape {arr.shape}"

        assert self.face.dtype in [np.int32, np.int64], \
            f"face must be int32 or int64, got {self.face.dtype}"
        assert self.contact_point.dtype in [np.float32, np.float64], \
            f"contact_point must be float32 or float64, got {self.contact_point.dtype}"
        assert self.angle.dtype in [np.float32, np.float64], \
            f"angle must be float32 or float64, got {self.angle.dtype}"
        assert self.push_distance.dtype in [np.float32, np.float64], \
            f"push_distance must be float32 or float64, got {self.push_distance.dtype}"

        n = len(self.face)
        assert len(self.contact_point) == n
        assert len(self.angle) == n
        assert len(self.push_distance) == n

        assert np.all((self.face >= 0) & (self.face <= 5)), \
            "face must be in {0, …, 5}"
        assert np.all((self.contact_point >= 0) & (self.contact_point <= 1)), \
            "contact_point must be in [0, 1]"
        assert np.all((self.angle >= 0) & (self.angle <= np.pi)), \
            "angle must be in [0, π]"
        assert np.all((self.push_distance >= 0) & (self.push_distance <= 0.1)), \
            "push_distance must be in [0, 0.1]"


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
        assert self.t_poses.shape == (self.action.nenvs, self.n_timesteps, 3), \
            f"t_poses must be (nenvs, n_timesteps, 3), got shape {self.t_poses.shape}"
        assert self.t_distances.shape == (self.action.nenvs, self.n_timesteps), \
            f"t_distances must be (nenvs, n_timesteps), got shape {self.t_distances.shape}"


# ── Environment ───────────────────────────────────────────────────────────────


@jax.jit
def step_single(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
) -> tuple[js.data.JaxSimModelData, dict[str, Any]]:
    return js.model.step(
        model=model,
        data=data,
        link_forces=None,
        joint_force_references=None,
    )


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0))
def step_parallel(
    model: js.model.JaxSimModel,
    data: js.data.JaxSimModelData,
) -> tuple[js.data.JaxSimModelData, dict[str, Any]]:
    return step_single(model=model, data=data)



class PushTEnv:
    """
    Parallel Push-T environment.

    State: (nenvs, 3) array of [x, y, theta] poses.
    """

    def __init__(self, nenvs: int, record_video: bool = False):
        self._nenvs = nenvs
        self._record_video = record_video
        self._poses: np.ndarray = np.zeros((nenvs, 3), dtype=np.float64)
        self._frames: list = []

        rod_sdf = rod.Sdf.load(pathlib.Path("assets/t_block.sdf"))
        rod_sdf.model.switch_frame_convention(
            frame_convention=rod.FrameConvention.Urdf, explicit_frames=True
        )
        model_sdf_string = rod_sdf.serialize(pretty=True)

        self._model = js.model.JaxSimModel.build_from_model_description(
            model_description=model_sdf_string,
            time_step=0.001,
        )

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
        row_dist = 0.3 * row_length
        x, y = jnp.meshgrid(
            jnp.linspace(-row_dist, row_dist, num=row_length),
            jnp.linspace(-row_dist, row_dist, num=row_length),
        )
        xy_coordinate = jnp.stack([x.flatten(), y.flatten()], axis=-1)
        self._xy_centers = np.array(xy_coordinate)
        self._data = data_batch_t0.replace(
            model=self._model,
            base_position=data_batch_t0.base_position.at[:, :2].set(xy_coordinate),
        )
        
        self.reset()
        
        if record_video:
            mjcf_string, assets = jaxsim.mujoco.ModelToMjcf.convert(
                self._model.built_from,
                cameras=jaxsim.mujoco.loaders.MujocoCamera.build_from_target_view(
                    camera_name="t_block_cam",
                    lookat=[0, 0, 0.1],
                    distance=2,
                    azimuth=150,
                    elevation=-30,
                ),
            )

            self._mj_model_helpers = [
                jaxsim.mujoco.MujocoModelHelper.build_from_xml(mjcf_description=mjcf_string, assets=assets) for _ in range(nenvs)
            ]
            self._recorder = jaxsim.mujoco.MujocoVideoRecorder(
                model=self._mj_model_helpers[0].model,
                data=[helper.data for helper in self._mj_model_helpers],
                fps=int(1 / self._model.time_step),
                width=320 * 2,
                height=240 * 2,
            )


    @property
    def nenvs(self) -> int:
        return self._nenvs

    def reset(self, seed: int = 0) -> np.ndarray:
        """
        Reset all environments to random orientations at the origin.
        Returns poses (nenvs, 3).
        """
        rng = np.random.default_rng(seed)
        self._poses = np.zeros((self._nenvs, 3), dtype=np.float64)
        self._poses[:, 2] = rng.uniform(-np.pi, np.pi, size=self._nenvs)
        self._frames = []
        
        # Compute quaternions for rotation around Z
        half_theta = self._poses[:, 2] / 2.0
        qw = np.cos(half_theta)
        qz = np.sin(half_theta)
        qx = np.zeros_like(qw)
        qy = np.zeros_like(qw)
        quats = np.stack([qw, qx, qy, qz], axis=-1)
        
        # Compute new base position with XY from centers and Z=0.0
        new_base_position = self._data.base_position.at[:, :2].set(self._xy_centers + self._poses[:, :2])
        new_base_position = new_base_position.at[:, 2].set(0.0)

        # Reset internal JaxSim data
        self._data = self._data.replace(
            model=self._model,
            base_position=new_base_position,
            base_quaternion=jnp.array(quats),
            base_linear_velocity=jnp.zeros_like(self._data._base_linear_velocity),
            base_angular_velocity=jnp.zeros_like(self._data._base_angular_velocity),
        )
        self._data = step_parallel(self._model, self._data)
        
        return self._poses.copy()

    def step(self, action: Action, n_sim_steps: int = 10) -> ActionResult:
        """
        Apply actions to all environments. Returns ActionResult.

        For now, ignores the push_distance value, and simply takes the action for n_sim_steps.
        """
        assert isinstance(action, Action), "action must be an Action"
        assert action.nenvs == self._nenvs, f"action.nenvs != self._nenvs ({action.nenvs} != {self._nenvs})"
        # TODO: removed n_sim_steps and use push_distance
        
        t_poses_list = []
        t_distances_list = []
        
        for _ in range(n_sim_steps):
            self._data = step_parallel(self._model, self._data)
            
            x = self._data.base_position[:, 0] - self._xy_centers[:, 0]
            y = self._data.base_position[:, 1] - self._xy_centers[:, 1]
            qw = self._data.base_orientation[:, 0]
            qz = self._data.base_orientation[:, 3]
            theta = 2.0 * jnp.arctan2(qz, qw)
            
            poses = jnp.stack([x, y, theta], axis=-1)
            t_poses_list.append(poses)
            t_distances_list.append(jnp.zeros(self._nenvs))
            
            if self._record_video:
                for helper, pos, quat, jpos in zip(
                    self._mj_model_helpers,
                    self._data.base_position,
                    self._data.base_orientation,
                    self._data.joint_positions,
                    strict=True,
                ):
                    helper.set_base_position(position=pos)
                    helper.set_base_orientation(orientation=quat)
                    if self._model.dofs() > 0:
                        helper.set_joint_positions(
                            positions=jpos, joint_names=self._model.joint_names()
                        )
                self._recorder.record_frame(camera_name="t_block_cam")
                
        t_poses = jnp.stack(t_poses_list, axis=1)
        t_distances = jnp.stack(t_distances_list, axis=1)
        
        return ActionResult(
            action=action,
            t_poses=np.array(t_poses),
            t_distances=np.array(t_distances),
        )

    @property
    def poses(self) -> np.ndarray:
        """Current poses (nenvs, 3) — read-only copy."""
        return self._poses.copy()

    def render(self) -> None:
        pass

    def save_video(self, filename: str, delete_cache_after_saving: bool = True) -> None:
        assert self._record_video, "record_video must be True to save video"
        assert len(self._recorder.frames) > 0, "No frames to save"
        
        imageio.mimwrite(filename, self._recorder.frames, fps=self._recorder.fps)
        if delete_cache_after_saving:
            self._recorder.frames.clear()
