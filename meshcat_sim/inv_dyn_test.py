# Copyright 2023 Inria
# SPDX-License-Identifier: BSD-2-Clause

"""
In this short script, we show how to compute inverse dynamics (RNEA), i.e. the
vector of joint torques corresponding to a given motion.
"""

from os.path import abspath, dirname, join
from pathlib import Path

import numpy as np
import pinocchio as pin


# Load the model from a URDF file
# Change to your own URDF file here, or give a path as command-line argument
pinocchio_model_dir = str(Path('./URDF/VREP_URDF').absolute())
# model_path = (pinocchio_model_dir + '/URDF4DynamicsSim.urdf')
mesh_dir = pinocchio_model_dir
# urdf_filename = "ur5_robot.urdf"
urdf_model_path = (pinocchio_model_dir + '/URDF4DynamicsSim.urdf')
model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=mesh_dir)

# Build a data frame associated with the model
data = model.createData()


# Sample a random joint configuration, joint velocities and accelerations
# q = np.array([1.0, 0.092, 1.0,  -0.0755, 1.0,  -0.1442, 1.0,   0., 1.0,   0., 1.0,   0., 1.0,  -0.2165, 1.0,
#               0.2599, 1.0,  -0.1241, 1.0,   0., 1.0,   0., 1.0,   0., 1.0,   0.1363, 1.0,   0.5418, 1.0,
#               -0.4709, 1.0,   0., 1.0,   0., 1.0,   0.])
# q = np.array([0.,  0.,  0.,  0.,  0.,  0.,  1.,
#               0., -0.3478,  0.1927,  0.,  0.,  0., -0.,
#               0.3504, -0.1942,  0.,  0.,  0.,  0.,  0.3477,
#               -0.1926,  0.,  0.,  0.])
q = np.array([0., -0.3478,  0.1927,  0.,  0.,  0., -0.,
              0.3504, -0.1942,  0.,  0.,  0.,  0.,  0.3477,
              -0.1926,  0.,  0.,  0.])
v = np.zeros(model.nv)  # in rad/s for the UR5
a = np.zeros(model.nv)  # in rad/s² for the UR5

# Computes the inverse dynamics (RNEA) for all the joints of the robot
tau = pin.rnea(model, data, q, v, a)

# Print out to the vector of joint torques (in N.m)
print("Joint torques: " + str(np.round(tau, 4)))
