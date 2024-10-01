import meshcat.geometry as g
import pinocchio as pin
from pathlib import Path
from pinocchio import JointModelFreeFlyer 
import numpy as np


class hex:
    # This class is supposed to provide the shits required to finish this stupid ass project

    def __init__(self, home_visualize=True) -> None:
        #Lets start by loading the URDF Files from the directory to visualizer
        # 1. We give the path to the URDF file (this serves as the blueprint for buliding the robot).
        # 2. Now we have to build the robot in the visualizer for which we have to provide the blueprint aka URDF
        #    file along with building components (.dae files) by providing the address to the dae file directory
        #    and lastly we need to mention if the base link is fixed or free-flyer or planar or prismatic.
        
        """Note:self.robot  an object created by Pinocchios RobotWrapper. It contains all the information 
           about the robot, including its links (rigid components), joints (the connections allowing motion), 
           and physical properties like mass and inertia.
           What self.robot Does:
            -It loads the robots structure from a URDF file (which describes how the robot is physically built).
            -It allows you to compute things like:
              * Forward kinematics: What are the positions of all the robots parts, given the joint angles.
              * Inverse kinematics: What joint angles are needed to place the robot's end effector at a certain 
                position.
              * Dynamics: How external forces and torques will affect the robots movement.
              
           Example Actions You Can Perform with self.robot:
             -self.robot.forwardKinematics(q): Compute the position of the robots links given 
              a joint configuration q.
             -self.robot.updateGeometryPlacements(): Update the visual placement of the robots
              parts after changing the joint configuration.
             -self.robot.model.getFrameId("foot"): Get the ID of a frame (such as the foot) to
              identify and control that specific part of the robot.
        """
        # 3. We have to find the forward kinematics of the robot using the function "self.robot.forwardKinematics(q)"
        #    to which we feed a vector of 43 elements: 
        #    [0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.
        #     0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0.]
        #    Here the first 7 suggest the home config of the base or root link (X_Y_Z_w_x_y_z(Quaternionangles)) and
        #    the remaining 36 vectors corresponds to the 18 joints exculding any fixed joint.
        self.urdf_filename=str('/home/jithin/Desktop/Jithin/HEXYSTUDY/URDF/URDF4Pin/URDF4Pin.urdf')
        Directory_path=str('/home/jithin/Desktop/Jithin/HEXYSTUDY/URDF/URDF4Pin')
        self.robot=pin.RobotWrapper.BuildFromURDF(self.urdf_filename,Directory_path,root_joint=JointModelFreeFlyer())
        self.robot.forwardKinematics(pin.neutral(self.robot.model))

        self.transformation_matrices = [] 
        for i, placement in enumerate(self.robot.data.oMi):
            R = placement.rotation  # 3x3 rotation matrix
            p = placement.translation  # 3x1 position vector
            T = np.eye(4)
            # Assign the rotation matrix and position vector
            T[:3, :3] = R
            T[:3, 3] = p
            # Append the transformation matrix to the list
            self.transformation_matrices.append(T)
            # Print the transformation matrix
            print(f"Joint {i} Transformation Matrix:")
            print(T)
            print()  # For better readability

def main():
    home_visualize=True
    hex_robot = hex(home_visualize)

if __name__==main():
    main()