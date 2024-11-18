from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from hexapod_v2_5 import hexapod
import numpy as np

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
q = hexy.compute_gait(v=0.5, STEPS=5, STEP_CNT=3)

# Step 2: Start the simulation
print("Starting the simulation...")
sim.startSimulation()


for q_i in q:
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
