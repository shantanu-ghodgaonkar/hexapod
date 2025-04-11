from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from hexapod_v2_5_3 import hexapod
import numpy as np
from time import sleep
import pinocchio as pin
from scipy.optimize import minimize


def get_pose(sim, camera_object: int, joint_objects: np.ndarray):
    """
    Retrieves the current pose of the robot and its joint positions as a configuration vector.

    Parameters:
    ----------
    sim : object
        The ZeroMQ remote API `sim` object from CoppeliaSim.
    camera_object : int
        Handle to the camera or reference frame object (e.g., robot base or IMU frame).
    joint_objects : np.ndarray
        A (6, 3) array of joint handles representing the hexapod's 18 joints.

    Returns:
    -------
    np.ndarray
        A 25-dimensional configuration vector `q` where:
            q[0:7]   -> robot base pose in XYZ + quaternion (world frame),
            q[7:25]  -> current joint angles for all 18 joints.
    """
    # Initialize configuration vector with 1s (will overwrite values)
    q = np.ones(25)

    # Get pose of the reference camera frame and find base pose
    q[:7] = np.round(pin.SE3ToXYZQUAT(pin.XYZQUATToSE3(
        sim.getObjectPose(camera_object)) * hexy.cMb_se3), 5)

    # Override Z-position (q[2]) to 0.0 for planar motion assumptions in controller
    q[2] = 0.0

    # Flatten joint object handles and fetch joint positions efficiently
    q[7:] = [sim.getJointPosition(int(h)) for h in joint_objects.flatten()]

    return q


hexy = hexapod(init_viz=False)

# Step 1: Connect to the CoppeliaSim Remote API server
client = RemoteAPIClient()
# sim = client.get_object('sim')
sim = client.getObject('sim')

robot_base = sim.getObject('/robot_base_respondable')
camera = sim.getObject('/CAM_respondable')

joint_objects = np.zeros((6, 3))

for i, j in np.ndindex(joint_objects.shape):
    obj = sim.getObject(f'/Revolute_joint_{j}{i+1}')
    print(f'Revolute_joint_{j}{i+1} has ID = {obj}')
    joint_objects[i][j] = obj
    sim.setJointPosition(obj, 0)

hexy.qc = get_pose(sim=sim, camera_object=camera, joint_objects=joint_objects)
hexy.state_c = hexy.forward_kinematics(hexy.qc)

# Step 2: Start the simulation
print("Starting the simulation...")
sim.startSimulation()
# sim.addLog(sim.verbosity_scriptinfos, f"Waiting for simulation to setup")
# sleep(5)
sim.addLog(sim.verbosity_scriptinfos, f"Simulation Started...")

WAYPOINTS = 20
# DIR = 'N'
# step_size_mult = 1
wp = hexy.generate_waypoints(WAYPOINTS=WAYPOINTS,
                             step_size_xy_mult=1, leg_set=0)
# hexy.plot_trajctory(state=wp.T, title='start half')
for wpi in wp.T:
    q_i = hexy.optimize(
        q0=hexy.qc, desired_pos=wpi.reshape(-1, 1)).x
    q_i[2] = sim.getObjectPose(robot_base)[2]

    sim.setObjectPose(robot_base, list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })

    hexy.qc = get_pose(sim=sim, camera_object=camera,
                       joint_objects=joint_objects)
    hexy.state_c = hexy.forward_kinematics(hexy.qc)

wp = hexy.generate_waypoints(WAYPOINTS=WAYPOINTS,
                             step_size_xy_mult=2, leg_set=1)
# hexy.plot_trajctory(state=wp.T, title='start half')
for wpi in wp.T:
    q_i = hexy.optimize(
        q0=hexy.qc, desired_pos=wpi.reshape(-1, 1)).x
    q_i[2] = sim.getObjectPose(robot_base)[2]

    sim.setObjectPose(robot_base, list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })

    hexy.qc = get_pose(sim=sim, camera_object=camera,
                       joint_objects=joint_objects)
    hexy.state_c = hexy.forward_kinematics(hexy.qc)


wp = hexy.generate_waypoints(WAYPOINTS=WAYPOINTS,
                             step_size_xy_mult=2, leg_set=0)
# hexy.plot_trajctory(state=wp.T, title='start half')
for wpi in wp.T:
    q_i = hexy.optimize(
        q0=hexy.qc, desired_pos=wpi.reshape(-1, 1)).x
    q_i[2] = sim.getObjectPose(robot_base)[2]

    sim.setObjectPose(robot_base, list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })

    hexy.qc = get_pose(sim=sim, camera_object=camera,
                       joint_objects=joint_objects)
    hexy.state_c = hexy.forward_kinematics(hexy.qc)


wp = hexy.generate_waypoints(WAYPOINTS=WAYPOINTS,
                             step_size_xy_mult=1, leg_set=1)
# hexy.plot_trajctory(state=wp.T, title='start half')
for wpi in wp.T:
    q_i = hexy.optimize(
        q0=hexy.qc, desired_pos=wpi.reshape(-1, 1)).x
    q_i[2] = sim.getObjectPose(robot_base)[2]

    sim.setObjectPose(robot_base, list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })

    hexy.qc = get_pose(sim=sim, camera_object=camera,
                       joint_objects=joint_objects)
    hexy.state_c = hexy.forward_kinematics(hexy.qc)

sleep(3)

sim.addLog(sim.verbosity_scriptinfos, f"Stopping Simulation...")
sim.stopSimulation()
