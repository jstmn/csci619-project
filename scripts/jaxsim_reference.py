import pathlib
import icub_models
import jax.numpy as jnp
import jaxsim.api as js

"""
python scripts/jaxsim_reference.py
"""

# Load the iCub model
model_path = icub_models.get_model_file("iCubGazeboV2_5")
joints = ('torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
          'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
          'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch',
          'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',
          'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch',
          'r_ankle_roll')

# Build and reduce the model
model_description = pathlib.Path(model_path)

full_model = js.model.JaxSimModel.build_from_model_description(
    model_description=model_description,
)

model = js.model.reduce(model=full_model, considered_joints=joints)

# Initialize model data
data = js.data.JaxSimModelData.build(model=model, base_position=jnp.array([0.0, 0.0, 1.0]))

# Frame and dynamics computations
frame_index = js.frame.name_to_idx(model=model, frame_name="l_foot")

# Frame transformation
W_H_F = js.frame.transform(model=model, data=data, frame_index=frame_index)

# Frame Jacobian
W_J_F = js.frame.jacobian(model=model, data=data, frame_index=frame_index)

# Dynamics properties
M = js.model.free_floating_mass_matrix(model=model, data=data)      # Mass matrix
h = js.model.free_floating_bias_forces(model=model, data=data)      # Bias forces
g = js.model.free_floating_gravity_forces(model=model, data=data)   # Gravity forces
C = js.model.free_floating_coriolis_matrix(model=model, data=data)  # Coriolis matrix

# Print dynamics results
print(f"{M.shape=} \n{h.shape=} \n{g.shape=} \n{C.shape=}")