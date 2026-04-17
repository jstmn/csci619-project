"""
Push-T parallel environment.

Block state: (x, y, theta) where (x, y) is the link-frame origin in world
coordinates and theta is the CCW rotation in radians.

Faces (A-F) go around the perimeter of the T; corners p0-p7 are at floor
level (z = 0) in the body frame defined in the URDF / README.
"""

import copy
from dataclasses import dataclass
import pathlib
import numpy as np
import functools
from typing import Any
from lxml import etree

import imageio
import jax
import jax.numpy as jnp
import jaxsim.api as js
import jaxsim.mujoco
import matplotlib.pyplot as plt
import mujoco as mj
import rod


T_RADIUS = 0.15
WORKSPACE_WIDTH = 1.5
WORKSPACE_HEIGHT = 1.5


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
    # push_distance:  np.ndarray

    @property
    def nenvs(self) -> int:
        return len(self.face)

    def __post_init__(self):
        for name, arr in (
            ("face",          self.face),
            ("contact_point", self.contact_point),
            ("angle",         self.angle),
            # ("push_distance", self.push_distance),
        ):
            assert isinstance(arr, np.ndarray), f"{name} must be np.ndarray"
            assert arr.shape == (self.nenvs, 1), f"{name} must be (nenvs, 1), got shape {arr.shape}"

        assert self.face.dtype in [np.int32, np.int64], \
            f"face must be int32 or int64, got {self.face.dtype}"
        assert self.contact_point.dtype in [np.float32, np.float64], \
            f"contact_point must be float32 or float64, got {self.contact_point.dtype}"
        assert self.angle.dtype in [np.float32, np.float64], \
            f"angle must be float32 or float64, got {self.angle.dtype}"
        # assert self.push_distance.dtype in [np.float32, np.float64], \
        #     f"push_distance must be float32 or float64, got {self.push_distance.dtype}"

        n = len(self.face)
        assert len(self.contact_point) == n
        assert len(self.angle) == n
        # assert len(self.push_distance) == n

        assert np.all((self.face >= 0) & (self.face <= 5)), \
            "face must be in {0, …, 5}"
        assert np.all((self.contact_point >= 0) & (self.contact_point <= 1)), \
            "contact_point must be in [0, 1]"
        assert np.all((self.angle >= 0) & (self.angle <= np.pi)), \
            "angle must be in [0, π]"
        # assert np.all((self.push_distance >= 0) & (self.push_distance <= 0.1)), \
        #     "push_distance must be in [0, 0.1]"


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

    def __init__(self, nenvs: int, record_video: bool = False, visualize: bool = False):
        self._nenvs = nenvs
        self._record_video = record_video
        self._visualize = visualize
        self._visualizer = None
        self._viewer = None
        self._poses: np.ndarray = np.zeros((nenvs, 3), dtype=np.float64)
        self._target_poses: np.ndarray = np.zeros((nenvs, 3), dtype=np.float64)
        self._frames: list = []

        rod_sdf = rod.Sdf.load(pathlib.Path("assets/scene.sdf"))
        for model in rod_sdf.models():
            model.switch_frame_convention(
                frame_convention=rod.FrameConvention.Urdf, explicit_frames=True
            )
        model_sdf_string = rod_sdf.serialize(pretty=True)

        self._model = js.model.JaxSimModel.build_from_model_description(
            model_description=model_sdf_string,
            time_step=0.001,
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

        if self._visualize:
            mjcf_string, assets = jaxsim.mujoco.ModelToMjcf.convert(self._model.built_from)
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
            self._visualizer = jaxsim.mujoco.MujocoVisualizer(
                model=self._mj_multi_model, data=self._mj_multi_data
            )
            self._viewer = self._visualizer.open_viewer()
            jaxsim.mujoco.MujocoVisualizer.setup_viewer_camera(
                self._viewer, lookat=[WORKSPACE_WIDTH/2, WORKSPACE_HEIGHT/2, 0.1], distance=3.0, azimuth=150, elevation=-30
            )
            self._sync_visualizer()
        else:
            self._visualizer = None
            self._viewer = None

        if record_video:
            mjcf_string, assets = jaxsim.mujoco.ModelToMjcf.convert(
                self._model.built_from,
                cameras=jaxsim.mujoco.loaders.MujocoCamera.build_from_target_view(
                    camera_name="t_block_cam",
                    lookat=[0.75, 0.75, 0.1],
                    distance=3.0,
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

    def _sync_visualizer(self) -> None:
        if self._visualizer is None:
            return
            
        for i in range(self._nenvs):
            # Update free joint for env_center_i
            jnt_id = mj.mj_name2id(self._mj_multi_model, mj.mjtObj.mjOBJ_JOINT, f"world_to_base_{i}")
            if jnt_id != -1:
                qpos_adr = self._mj_multi_model.jnt_qposadr[jnt_id]
                self._mj_multi_data.qpos[qpos_adr : qpos_adr + 3] = self._data.base_position[i]
                self._mj_multi_data.qpos[qpos_adr + 3 : qpos_adr + 7] = self._data.base_orientation[i]
            
            # Update 1D joints
            for jnt_name in self._model.joint_names():
                multi_jnt_name = f"{jnt_name}_{i}"
                multi_jnt_id = mj.mj_name2id(self._mj_multi_model, mj.mjtObj.mjOBJ_JOINT, multi_jnt_name)
                if multi_jnt_id != -1:
                    qpos_adr = self._mj_multi_model.jnt_qposadr[multi_jnt_id]
                    orig_jnt_idx = self._model.joint_names().index(jnt_name)
                    self._mj_multi_data.qpos[qpos_adr] = self._data.joint_positions[i, orig_jnt_idx]
                    
        self._visualizer.sync(self._viewer)

    def reset(self, seed: int = 0) -> np.ndarray:
        """
        Reset all environments to random orientations at the origin.
        Returns poses (nenvs, 3).
        """
        rng = np.random.default_rng(seed)

        # First, set randomized poses for the T and target T
        self._target_poses = np.zeros((self._nenvs, 3), dtype=np.float64)
        self._target_poses[:, 0] = rng.uniform(T_RADIUS, WORKSPACE_WIDTH - T_RADIUS, size=self._nenvs)
        self._target_poses[:, 1] = rng.uniform(T_RADIUS, WORKSPACE_HEIGHT - T_RADIUS, size=self._nenvs)
        self._target_poses[:, 2] = rng.uniform(-np.pi, np.pi, size=self._nenvs)
        # 
        self._poses = np.zeros((self._nenvs, 3), dtype=np.float64)
        self._poses[:, 0] = rng.uniform(T_RADIUS, WORKSPACE_WIDTH - T_RADIUS, size=self._nenvs)
        self._poses[:, 1] = rng.uniform(T_RADIUS, WORKSPACE_HEIGHT - T_RADIUS, size=self._nenvs)
        self._poses[:, 2] = rng.uniform(-np.pi, np.pi, size=self._nenvs)

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
        new_joint_positions = new_joint_positions.at[:, self._T_x_idx].set(self._poses[:, 0])
        new_joint_positions = new_joint_positions.at[:, self._T_y_idx].set(self._poses[:, 1])
        new_joint_positions = new_joint_positions.at[:, self._T_theta_idx].set(self._poses[:, 2])
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
        self._data = step_parallel(self._model, self._data)
        self._sync_visualizer()
        
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
            
            x = self._data.joint_positions[:, self._T_x_idx]
            y = self._data.joint_positions[:, self._T_y_idx]
            theta = self._data.joint_positions[:, self._T_theta_idx]
            
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
                
            if self._visualizer is not None:
                self._sync_visualizer()
                import time
                time.sleep(self._model.time_step)
                
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
