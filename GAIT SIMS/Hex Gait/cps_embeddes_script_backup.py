import numpy as np
from time import sleep


def sysCall_init():
    sim = require('sim')

    # do some initialization here
    #
    # Instead of using globals, you can do e.g.:
    # self.myVariable = 21000000
    joint_objects = np.zeros((6, 3))

    for i, j in np.ndindex(joint_objects.shape):
        obj = sim.getObject(f'/Revolute_joint_{j}{i+1}')
        print(f'Revolute_joint_{j}{i+1} has ID = {obj}')
        joint_objects[i][j] = obj
        sim.setJointPosition(obj, 0)

#    sim.addLog(sim.verbosity_scriptinfos, f"Resetting robot state.")
#    q = np.array([1 if i == 6 else 0 for i in range(25)], dtype=np.float64)
#    q[2] = sim.getObjectPose(sim.getObject('/HexY'))[2]
#    sim.setObjectPose(sim.getObject('/HexY'), list(q[:7]))
#    sim.moveToConfig_init({
#        'joints': list(joint_objects.flatten()),
#        'targetPos': list(q[7:])
#    })

    sim.stopSimulation()


def sysCall_actuation():
    # put your actuation code here
    pass


def sysCall_sensing():
    # put your sensing code here
    pass


def sysCall_cleanup():
    # do some clean-up here
    pass

# See the user manual or the available code snippets for additional callback functions and details
