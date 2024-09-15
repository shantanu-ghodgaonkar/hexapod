# we import useful libraries
from time import sleep, time
import numpy as np
import pinocchio as pin
import os
import meshcat
import matplotlib as mp
import matplotlib.pyplot as plt
import scipy
from pathlib import Path
import sys

import scipy.optimize

# we don't want to print every decimal!
np.set_printoptions(suppress=True, precision=4)

PI = 3.141592


class hexapod:

    def __init__(self) -> None:
        self.pin_model_dir = str(
            Path('URDF/URDF4Pin').absolute())
        # self.pin_model_dir = str(
        #     Path('/home/shantanu/Documents/hexapod_files/CS_Hexapod_Ex_1').absolute())
        # self.pin_model_dir = str(Path('re2').absolute())

        # You should change here to set up your own URDF file or just pass it as an argument of this example.
        self.urdf_filename = (self.pin_model_dir + '/URDF4Pin.urdf')
        # self.urdf_filename = (self.pin_model_dir + '/CS_Hexapod_Ex_1.urdf')
        # self.urdf_filename = (self.pin_model_dir + '/IKTRials_redone naming convention.urdf')

        self.robot = pin.RobotWrapper.BuildFromURDF(
            self.urdf_filename, self.pin_model_dir, root_joint=pin.JointModelFreeFlyer())

        self.robot.forwardKinematics(pin.neutral(self.robot.model))
        self.robot.updateGeometryPlacements()
        # self.print_all_IDs()

        # self.robot.model.addFrame(pin.Frame())

        # for i in range(1, 6):
        #     frame_name = "Foot_" + str(i)
        #     self.robot.model.addFrame(
        #         pin.Frame(frame_name, 1, 1)
        #     )
        print("DEBUG POINT")

    def foot_pos_err(self, q, FRAME_ID=9, desired_pos=np.zeros(3)):
        self.robot.forwardKinematics(q)
        # current_pos = self.robot.visual_model.geometryObjects.tolist()[
        #     (FOOT) * 4].placement.translation
        current_pos = self.robot.framePlacement(q, FRAME_ID).translation
        error = current_pos - desired_pos
        return error.dot(error)

    def print_all_IDs(self):
        for frame in self.robot.model.frames:
            # Print the frame name and type
            print(
                f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.robot.model.q0, self.robot.model.getFrameId(frame.name))}")

    def inverse_geometery(self, FRAME_ID=9, desired_pos=np.zeros(3)):
        bounds = [(0., 0.)]*self.robot.nq
        bounds[0] = (0, 0)
        bounds[1] = (0, 0)
        bounds[2] = (0, 0)
        bounds[3] = (0, 0)
        bounds[4] = (0, 0)
        bounds[5] = (0, 0)
        bounds[6] = (1, 1)
        for i in range(7, self.robot.nq):
            bounds[i] = ((-PI/3), (PI/3))
        res = scipy.optimize.minimize(
            hexy.foot_pos_err, self.robot.nqs, args=(FRAME_ID, desired_position), bounds=bounds)


if __name__ == "__main__":
    hexy = hexapod()

    try:
        viz = pin.visualize.MeshcatVisualizer(
            hexy.robot.model, hexy.robot.collision_model, hexy.robot.visual_model)
        viz.initViewer(open=True)
        # model.apply_colors(viz)
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install Python meshcat"
        )
        print(err)
        sys.exit(0)

    # Load the robot in the viewer.
    viz.loadViewerModel()

    viz.displayFrames(visibility=True)
    viz.displayCollisions(visibility=False)

    q0 = pin.neutral(hexy.robot.model)
    viz.display(q0)

    desired_position = np.array([-0.020, -1.375, 0.1])
    hexy.robot.forwardKinematics(res.x)
    viz.display(res.x)
    print("DEBUG POINT")
