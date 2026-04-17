"""
python scripts/jaxsim_reference.py

Note that this script is derived from https://github.com/gbionics/jaxsim/blob/main/examples/jaxsim_as_physics_engine_advanced.ipynb
"""

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
from jaxsim import logging
from lxml import etree
from rod.builder.primitives import SphereBuilder

logging.set_logging_level(logging.LoggingLevel.WARNING)
print(f"Running on {jax.devices()}")

# ── Build model description ────────────────────────────────────────────────────

rod_sdf = rod.Sdf(
    version="1.7",
    model=SphereBuilder(radius=0.10, mass=1.0, name="sphere")
    .build_model()
    .add_link()
    .add_inertial()
    .add_visual()
    .add_collision()
    .build(),
)
rod_sdf.model.switch_frame_convention(frame_convention=rod.FrameConvention.Urdf, explicit_frames=True)
model_sdf_string = rod_sdf.serialize(pretty=True)

os.environ["JAXSIM_COLLISION_SPHERE_POINTS"] = "50"

# ── Create model and data ──────────────────────────────────────────────────────

model = js.model.JaxSimModel.build_from_model_description(
    model_description=model_sdf_string,
    time_step=0.001,
)
data_single = js.data.JaxSimModelData.zero(model=model)

T = jnp.arange(start=0, stop=1.0, step=model.time_step)

# ── Generate batched initial data ─────────────────────────────────────────────

key = jax.random.PRNGKey(seed=0)
batch_size = 9
row_length = int(jnp.sqrt(batch_size))
row_dist = 0.3 * row_length
key, *subkeys = jax.random.split(key=key, num=batch_size + 1)

data_batch_t0 = jax.vmap(
    lambda key: js.data.random_model_data(
        model=model,
        key=key,
        base_pos_bounds=([0, 0, 0.3], [0, 0, 1.2]),
        base_vel_lin_bounds=(0, 0),
        base_vel_ang_bounds=(0, 0),
    )
)(jnp.vstack(subkeys))

x, y = jnp.meshgrid(
    jnp.linspace(-row_dist, row_dist, num=row_length),
    jnp.linspace(-row_dist, row_dist, num=row_length),
)
xy_coordinate = jnp.stack([x.flatten(), y.flatten()], axis=-1)
data_batch_t0 = data_batch_t0.replace(
    model=model,
    base_position=data_batch_t0.base_position.at[:, :2].set(xy_coordinate),
)
print("W_p_B(t0)=\n", data_batch_t0.base_position[0:10])

# ── Define step functions ──────────────────────────────────────────────────────


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


# Warm up JIT compilation.
_ = step_single(model, data_single)
_ = step_parallel(model, data_batch_t0)

# ── MuJoCo setup ──────────────────────────────────────────────────────────────

mjcf_string, assets = jaxsim.mujoco.ModelToMjcf.convert(
    model.built_from,
    cameras=jaxsim.mujoco.loaders.MujocoCamera.build_from_target_view(
        camera_name="sphere_cam",
        lookat=[0, 0, 0.3],
        distance=4,
        azimuth=150,
        elevation=-10,
    ),
)

mj_model_helpers = [
    jaxsim.mujoco.MujocoModelHelper.build_from_xml(mjcf_description=mjcf_string, assets=assets)
    for _ in range(batch_size)
]

recorder = jaxsim.mujoco.MujocoVideoRecorder(
    model=mj_model_helpers[0].model,
    data=[helper.data for helper in mj_model_helpers],
    fps=int(1 / model.time_step),
    width=320 * 2,
    height=240 * 2,
)

# Build a combined MuJoCo model with all batch_size spheres for the passive viewer.
# Each free body contributes 7 entries to qpos: [x, y, z, qw, qx, qy, qz].
_colors = [
    "0.8 0.3 0.3 1",
    "0.3 0.8 0.3 1",
    "0.3 0.3 0.8 1",
    "0.8 0.8 0.3 1",
    "0.8 0.3 0.8 1",
    "0.3 0.8 0.8 1",
    "0.9 0.5 0.2 1",
    "0.5 0.2 0.9 1",
    "0.2 0.9 0.5 1",
]
_root = etree.Element("mujoco", model="multi_sphere")
etree.SubElement(_root, "option", gravity="0 0 -9.81")
_worldbody = etree.SubElement(_root, "worldbody")
etree.SubElement(_worldbody, "geom", name="floor", type="plane", size="10 10 0.1", rgba=".9 .9 .9 1")
for _i in range(batch_size):
    _body = etree.SubElement(_worldbody, "body", name=f"sphere_{_i}", pos="0 0 1")
    etree.SubElement(_body, "freejoint", name=f"joint_{_i}")
    etree.SubElement(_body, "geom", type="sphere", size="0.1", rgba=_colors[_i % len(_colors)])

mj_multi_model = mj.MjModel.from_xml_string(etree.tostring(_root, pretty_print=True).decode())
mj_multi_data = mj.MjData(mj_multi_model)

# Seed initial positions from data_batch_t0.
for _i, (_pos, _quat) in enumerate(zip(data_batch_t0.base_position, data_batch_t0.base_orientation)):
    mj_multi_data.qpos[_i * 7 : _i * 7 + 3] = _pos
    mj_multi_data.qpos[_i * 7 + 3 : _i * 7 + 7] = _quat

visualizer = jaxsim.mujoco.MujocoVisualizer(model=mj_multi_model, data=mj_multi_data)

# ── Run parallel simulation + visualize ───────────────────────────────────────

data = data_batch_t0
data_trajectory_list = []

print("Opening realtime viewer (all instances)...")
with visualizer.open(
    lookat=[0, 0, 0.3],
    distance=4,
    azimuth=150,
    elevation=-10,
) as viewer:
    for _ in T:
        data = step_parallel(model, data)
        data_trajectory_list.append(data)

        # Update video recorder helpers.
        for helper, pos, quat, jpos in zip(
            mj_model_helpers,
            data.base_position,
            data.base_orientation,
            data.joint_positions,
            strict=True,
        ):
            helper.set_base_position(position=pos)
            helper.set_base_orientation(orientation=quat)
            if model.dofs() > 0:
                helper.set_joint_positions(positions=jpos, joint_names=model.joint_names())
        recorder.record_frame(camera_name="sphere_cam")

        # Update combined model and sync passive viewer.
        for _i, (_pos, _quat) in enumerate(zip(data.base_position, data.base_orientation)):
            mj_multi_data.qpos[_i * 7 : _i * 7 + 3] = _pos
            mj_multi_data.qpos[_i * 7 + 3 : _i * 7 + 7] = _quat
        visualizer.sync(viewer)
        time.sleep(model.time_step)

# ── Save outputs ───────────────────────────────────────────────────────────────

imageio.mimwrite("sphere_simulation.mp4", recorder.frames, fps=recorder.fps)
print(f"Video saved to sphere_simulation.mp4 ({len(recorder.frames)} frames)")

data_trajectory = jax.tree.map(lambda *leafs: jnp.stack(leafs), *data_trajectory_list)
print(f"W_p_B: shape={data_trajectory.base_position.shape}")

plt.plot(T, data_trajectory.base_position[:, :, 2])
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.title("Height trajectory of the sphere")
plt.savefig("sphere_trajectory.png", dpi=150)
plt.show()
