import numpy as np
import matplotlib.pyplot as plt
import jiminy_py.core as jiminy  # The main module of jiminy - this is what gives access to the Robot
from jiminy_py.simulator import Simulator
from pathlib import Path

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


data_root_path = str(str(Path('jiminy').absolute()) + '/data/toys_models/simple_pendulum')
urdf_path = str(data_root_path + "/simple_pendulum.urdf")

# Instantiate and initialize the robot
robot = jiminy.Robot()
robot.initialize(urdf_path, mesh_package_dirs=[str(data_root_path)])

# Add a single motor
motor = jiminy.SimpleMotor("PendulumJoint")
robot.attach_motor(motor)
motor.initialize("PendulumJoint")

# Define the command: for now, the motor is off and doesn't modify the output torque.
def compute_command(t, q, v, sensor_measurements, command):
    command[:] = 0.0

# Instantiate and initialize the controller
robot.controller = jiminy.FunctionalController(compute_command)

# Create a simulator using this robot and controller
simulator = Simulator(robot)

# Set initial condition and simulation length
q0, v0 = np.array([0.1]), np.array([0.0])
simulation_duration = 10.0

# # Launch the simulation
# simulator.simulate(simulation_duration, q0, v0)

# # Get dictionary of logged scalar variables
# log_vars = simulator.log_data['variables']

# plt.plot(log_vars['Global.Time'], log_vars['currentPositionPendulum'])
# plt.title('Pendulum angle (rad)')
# plt.grid()
# plt.show()

camera_pose = ([5.0, 0.0, 2.0e-5], [np.pi/2, 0.0, np.pi/2])
# simulator.replay(camera_pose=camera_pose, backend="panda3d")

Kp = 5000
Kd = 0.05

# Define a new controller with Proportional-Derivative command
def compute_command(t, q, v, sensor_measurements, command):
    command[:] = - Kp * (q + Kd * v)

robot.controller = jiminy.FunctionalController(compute_command)

# Apply a force of 500N in the Y direction between t = 2.5 and t = 3s
def force_profile(t, q, v, f):
    if t > 2.5 and t < 3:
        f[1] = 200.0
    else:
        f[1] = 0.0

# Apply this force profile to a given frame.
simulator.register_profile_force("PendulumMass", force_profile)
simulator.simulate(simulation_duration, q0, v0)

# Get dictionary of logged scalar variables
log_vars = simulator.log_data['variables']

# Replay the simulation with new controller and external forces
simulator.replay(camera_pose=camera_pose)

plt.plot(log_vars['Global.Time'], log_vars['currentPositionPendulum'])
plt.title('Pendulum angle (rad)')
plt.grid()
plt.show()