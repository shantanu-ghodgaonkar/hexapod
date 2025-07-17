from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from hexapod_v2_5 import hexapod
import numpy as np
from time import sleep


# Step 1: Connect to the CoppeliaSim Remote API server
client = RemoteAPIClient()
# sim = client.get_object('sim')
sim = client.getObject('sim')

joint_objects = np.zeros((6, 3))

for i, j in np.ndindex(joint_objects.shape):
    obj = sim.getObject(f'/Revolute_joint_{j}{i+1}')
    print(f'Revolute_joint_{j}{i+1} has ID = {obj}')
    joint_objects[i][j] = obj
    sim.setJointPosition(obj, 0)

hexy = hexapod(init_viz=False)
v = 0.5
WAYPOINTS = 5

# Step 2: Start the simulation
print("Starting the simulation...")
sim.startSimulation()
# sim.addLog(sim.verbosity_scriptinfos, f"Waiting for simulation to setup")
# sleep(5)
sim.addLog(sim.verbosity_scriptinfos, f"Simulation Started...\nMoving North")
q_traj = hexy.compute_gait(v=v, WAYPOINTS=WAYPOINTS, STEP_CNT=1, DIR='N')
q_traj[:, 2] = [sim.getObjectPose(sim.getObject('/HexY'))[2]]*(q_traj.shape[0])
for q_i in q_traj:
    sim.setObjectPose(sim.getObject('/HexY'), list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })
# sleep(3)
sim.addLog(sim.verbosity_scriptinfos, f"Moving South")
q_traj = hexy.compute_gait(v=v, WAYPOINTS=WAYPOINTS, STEP_CNT=1, DIR='S')
q_traj[:, 2] = [sim.getObjectPose(sim.getObject('/HexY'))[2]]*(q_traj.shape[0])
for q_i in q_traj:
    sim.setObjectPose(sim.getObject('/HexY'), list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })
# sleep(3)

sim.addLog(sim.verbosity_scriptinfos, f"Moving West")
q_traj = hexy.compute_gait(v=v, WAYPOINTS=WAYPOINTS, STEP_CNT=1, DIR='W')
q_traj[:, 2] = [sim.getObjectPose(sim.getObject('/HexY'))[2]]*(q_traj.shape[0])
for q_i in q_traj:
    sim.setObjectPose(sim.getObject('/HexY'), list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })
# sleep(3)

sim.addLog(sim.verbosity_scriptinfos, f"Moving East")
q_traj = hexy.compute_gait(v=v, WAYPOINTS=WAYPOINTS, STEP_CNT=1, DIR='E')
q_traj[:, 2] = [sim.getObjectPose(sim.getObject('/HexY'))[2]]*(q_traj.shape[0])
for q_i in q_traj:
    sim.setObjectPose(sim.getObject('/HexY'), list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })

sim.addLog(sim.verbosity_scriptinfos, f"Moving Northeast")
q_traj = hexy.compute_gait(v=v, WAYPOINTS=WAYPOINTS, STEP_CNT=1, DIR='NE')
q_traj[:, 2] = [sim.getObjectPose(sim.getObject('/HexY'))[2]]*(q_traj.shape[0])
for q_i in q_traj:
    sim.setObjectPose(sim.getObject('/HexY'), list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })

sim.addLog(sim.verbosity_scriptinfos, f"Moving Southeast")
q_traj = hexy.compute_gait(v=v, WAYPOINTS=WAYPOINTS, STEP_CNT=1, DIR='SE')
q_traj[:, 2] = [sim.getObjectPose(sim.getObject('/HexY'))[2]]*(q_traj.shape[0])
for q_i in q_traj:
    sim.setObjectPose(sim.getObject('/HexY'), list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })

sim.addLog(sim.verbosity_scriptinfos, f"Moving Northwest")
q_traj = hexy.compute_gait(v=v, WAYPOINTS=WAYPOINTS, STEP_CNT=1, DIR='NW')
q_traj[:, 2] = [sim.getObjectPose(sim.getObject('/HexY'))[2]]*(q_traj.shape[0])
for q_i in q_traj:
    sim.setObjectPose(sim.getObject('/HexY'), list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })

sim.addLog(sim.verbosity_scriptinfos, f"Moving Southwest")
q_traj = hexy.compute_gait(v=v, WAYPOINTS=WAYPOINTS, STEP_CNT=1, DIR='SW')
q_traj[:, 2] = [sim.getObjectPose(sim.getObject('/HexY'))[2]]
for q_i in q_traj:
    sim.setObjectPose(sim.getObject('/HexY'), list(q_i[:7]))
    sim.moveToConfig({
        'joints': list(joint_objects.flatten()),
        'targetPos': list(q_i[7:])
    })

sim.addLog(sim.verbosity_scriptinfos, f"Resetting robot state.")
q = np.array([1 if i == 6 else 0 for i in range(
    hexy.robot.nq)], dtype=np.float64)
q[2] = sim.getObjectPose(sim.getObject('/HexY'))[2]
sim.setObjectPose(sim.getObject('/HexY'), list(q[:7]))
sim.moveToConfig({
    'joints': list(joint_objects.flatten()),
    'targetPos': list(q[7:])
})

# Step 4: Stop the simulation
sim.addLog(sim.verbosity_scriptinfos, f"Stopping Simulation...")
sim.stopSimulation()
