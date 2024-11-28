from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from hexapod_v2_5 import hexapod
import numpy as np
from pathlib import Path
from time import strftime

# Step 1: Connect to the CoppeliaSim Remote API server
client = RemoteAPIClient()
# sim = client.get_object('sim')
sim = client.getObject('sim')

joint_objects = np.zeros((6, 3))

for i, j in np.ndindex(joint_objects.shape):
    obj = sim.getObject(f'/Revolute_joint_{j}{i+1}')
    print(f'Revolute_joint_{j}{i+1} has ID = {obj}')
    joint_objects[i][j] = obj


hexy = hexapod(init_viz=False)
q_traj = hexy.compute_gait(v=0.5, STEPS=10, STEP_CNT=3)
# q_wps = np.vstack((
#     hexy.body_inverse_geometry(hexy.robot.q0, np.array([0, 0, 0, 0, 0, 0, 1])),
#     hexy.body_inverse_geometry(
#         hexy.robot.q0, np.array([0, 0, np.pi/3, 0, 0, 0, 1])),
#     hexy.body_inverse_geometry(hexy.robot.q0, np.array([0, 0, 0, 0, 0, 0, 1]))
# ))

# t_init = 0
# t_goal = 2
# dt = 0.01
# q_traj = np.zeros(
#     (int(((t_goal - t_init) / dt) * q_wps.shape[0]), q_wps.shape[1]))
# j = 0
# for i in range(0, q_wps.shape[0]-1):
#     t = 0
#     while t < t_goal:
#         # Compute trajectory using linear interpolation
#         q_traj[j, :] = hexy.compute_trajectory_p(
#             q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
#         t = (t + dt)
#         j += 1
# q_traj[j, :] = q_wps[-1]
# q_traj = q_traj[:j+1, :]

# gait_angles_file_path = Path(
#     f'gait_angles/gait_angles_pushup_{strftime("%Y%m%d_%H%M%S")}.npy')
# gait_angles_file_path.parent.mkdir(parents=True, exist_ok=True)
# np.save(gait_angles_file_path, q_traj)

# for i in range(6, 24, 3):
#     q[:, i+1] = -q[:, i+1]
#     q[:, i+2] = -q[:, i+2]

# hexy.viz.display(q_traj)

# Step 2: Start the simulation
print("Starting the simulation...")
sim.startSimulation()
q_traj[:, 2] = [sim.getObjectPose(sim.getObject('/HexY'))[2]]*(q_traj.shape[0])
for q_i in q_traj:
    sim.setObjectPose(sim.getObject('/HexY'), list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })
# Step 3: Pause for 5 seconds to let the simulation run
# sleep(5)

# sim.setJointPosition(joint_objects[0, 1], -1)
# sim.setJointPosition(joint_objects[2, 1], -1)
# sim.setJointPosition(joint_objects[4, 1], -1)
# for i, j in np.ndindex(joint_objects.shape):
#     print(
#         f'Revolute_joint_{j}{i+1} has Joint Force = {sim.getJointForce(joint_objects[i, j])}')

# for i, j in np.ndindex(joint_objects.shape):
#     if ((tuple((i, j)) == (0, 1))):
#         sim.setJointPosition(joint_objects[i, j], -1)
#     elif ((tuple((i, j)) == (2, 1)) or (tuple((i, j)) == (4, 1))):
#         sim.setJointPosition(joint_objects[i, j], +1)
#     else:
#         sim.setJointPosition(joint_objects[i, j], 0)
#     print(
#         f'Revolute_joint_{j}{i+1} has Joint Force = {np.round(sim.getJointForce(joint_objects[i, j]), 3)}Nm at Angle = {sim.getJointPosition(joint_objects[i, j])}')

# sleep(5)

# Step 4: Stop the simulation
print("Stopping the simulation...")
sim.stopSimulation()


# import numpy as np

# joint_objects = np.zeros((6, 3))

# def sysCall_init():
#     sim = require('sim')

#     # do some initialization here
#     #
#     # Instead of using globals, you can do e.g.:
#     # self.myVariable = 21000000
#     for i, j in np.ndindex(joint_objects.shape):
#         obj = sim.getObject(f'/Revolute_joint_{j}{i+1}')
#         print(f'Revolute_joint_{j}{i+1} has ID = {obj}')
#         joint_objects[i][j] = obj
# #    sim.setJointPosition(joint_objects[0, 1], -1)
# #    sim.setJointPosition(joint_objects[2, 1], -1)
# #    sim.setJointPosition(joint_objects[4, 1], -1)
# #    for i, j in np.ndindex(joint_objects.shape):
# #        print(f'Revolute_joint_{j}{i+1} has Joint Force = {sim.getJointForce(joint_objects[i, j])}')


# def sysCall_actuation():
#     # put your actuation code here

# #    sim.setJointPosition(joint_objects[0, 1], -1)
# #    sim.setJointPosition(joint_objects[2, 1], -1)
# #    sim.setJointPosition(joint_objects[4, 1], -1)
#     for i, j in np.ndindex(joint_objects.shape):
#         if ((tuple((i, j)) == (0, 1))):
#             sim.setJointPosition(joint_objects[i, j], -1)
#         elif ((tuple((i, j)) == (2, 1)) or (tuple((i, j)) == (4, 1))):
#             sim.setJointPosition(joint_objects[i, j], +1)
#         else:
#             sim.setJointPosition(joint_objects[i, j], 0)
#         print(f'Revolute_joint_{j}{i+1} has Joint Force = {np.round(sim.getJointForce(joint_objects[i, j]), 3)}Nm at Angle = {sim.getJointPosition(joint_objects[i, j])}')
#     sim.stopSimulation()
# #    pass

# def sysCall_sensing():
#     # put your sensing code here
#     pass

# def sysCall_cleanup():
#     # do some clean-up here
#     pass

# # See the user manual or the available code snippets for additional callback functions and details
