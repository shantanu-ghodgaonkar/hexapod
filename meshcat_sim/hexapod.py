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
        self.direction_slope = 0.0
        self.STEP_SIZE_XY = 0.05
        self.STEP_SIZE_Z = 0.025
        self.FOOT_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "foot" in frame.name]
        self.SHOULDER_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "Revolute_joint_0" in frame.name]
        self.state_flag = 'START'
        self.FOOT1HT = self.robot.framePlacement(self.robot.q0, 9)
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
        # current_pos = (np.concatenate(
        #     (current_pos, [1])) * self.FOOT1HT)[0:3, 3]
        error = current_pos - desired_pos
        return error.dot(error)

    def print_all_frame_info(self):
        for frame in self.robot.model.frames:
            # Print the frame name and type
            print(
                f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.robot.q0, self.robot.model.getFrameId(frame.name))}")

    def find_frame_info_by_name(self, name: str):
        for frame in self.robot.model.frames:
            if name in frame.name:
                print(
                    f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.robot.q0, self.robot.model.getFrameId(frame.name))}")

    def inverse_geometery(self, q, FRAME_ID=9, desired_pos=np.zeros(3)):
        bounds = [(0., 0.)]*self.robot.nq
        bounds[0] = (0, 0)
        bounds[1] = (0, 0)
        bounds[2] = (0, 0)
        bounds[3] = (0, 0)
        bounds[4] = (0, 0)
        bounds[5] = (0, 0)
        bounds[6] = (1, 1)
        for i in range(7, self.robot.nq):
            bounds[i] = ((-np.pi/3), (np.pi/3))
        # for i in range(7, self.robot.nq):
        #     if i % 2 == 0:
        #         bounds[i] = ((-np.pi/3), (np.pi/3))
        #     else:
        #         bounds[i] = (1, 1)
        res = scipy.optimize.minimize(
            self.foot_pos_err, q, args=(FRAME_ID, desired_pos), bounds=bounds)
        return res.x

    def north_vector(self):
        print("North Vector")
        # Get frame of Joint_01
        p1 = self.robot.framePlacement(self.robot.q0, 3).translation
        # Get frame for Joint_02
        p2 = self.robot.framePlacement(self.robot.q0, 11).translation
        # find vector
        v = np.array([-(p2 - p1)[1], (p2 - p1)[0]])
        return v/(np.sqrt((v[0]**2) + (v[1]**2)))

    def south_vector(self):
        pass

    def east_vector(self):
        pass

    def west_vector(self):
        pass

    def north_east_vector(self):
        pass

    def north_west_vector(self):
        pass

    def south_east_vector(self):
        pass

    def south_west_vector(self):
        pass

    def default_vector(self):
        raise KeyError(
            "NOT A VAILD DIRECTION. CHOOSE ONE OF {N, S, E, W, NE, NW, SE, SW} ONLY")

    def generate_direction_vector(self, DIR='N'):
        slope_switch_dict = {
            'N': self.north_vector,
            'S': self.south_vector,
            'E': self.east_vector,
            'W': self.west_vector,
            'NE': self.north_east_vector,
            'NW': self.north_west_vector,
            'SE': self.south_east_vector,
            'SW': self.south_west_vector
        }
        return slope_switch_dict.get(DIR, self.default_vector)()

    def generate_path_function(self, DIR='N', FOOT_ID=9):
        v = self.generate_direction_vector(DIR=DIR)
        p1 = self.robot.framePlacement(self.robot.q0, FOOT_ID).translation[0:2]
        p2 = p1 + (self.STEP_SIZE_XY * v)
        print(f"v = {v}\np1 = {p1}\np2 = {p2}")
        # def x_t(t): return ((1 - t) * p1[0]) + (t * p2[0])
        self.x_t = lambda t: ((1 - t) * p1[0]) + (t * p2[0])
        self.y_t = lambda t:  ((1 - t) * p1[1]) + (t * p2[1])

        self.z_t = lambda t: (((-self.STEP_SIZE_Z)/0.25) *
                              (t ** 2)) + ((self.STEP_SIZE_Z / 0.25) * t)
        print("DEBUG POINT")

    def compute_trajectory(self, position_init, position_goal, t_init, t_goal, t):
        t_tot = t_goal - t_init
        self.desired_position = position_init + (
            ((10 / (t_tot**3)) * ((t - t_init)**3))
            + (((-15) / (t_tot**4)) * ((t - t_init)**4))
            + ((6 / (t_tot**5)) * ((t - t_init)**5))) * (position_goal - position_init)
        self.desired_velocity = (
            ((30 / (t_tot**3)) * ((t - t_init)**2))
            + (((-60) / (t_tot**4)) * ((t - t_init)**3))
            + ((30 / (t_tot**5)) * ((t - t_init)**4))) * (position_goal - position_init)
        self.desired_acceleration = (
            ((60 / (t_tot**3)) * (t - t_init))
            + (((-180) / (t_tot**4)) * ((t - t_init)**2))
            + ((120 / (t_tot**5)) * ((t - t_init)**3))) * (position_goal - position_init)

        return self.desired_position, self.desired_acceleration, self.desired_acceleration


if __name__ == "__main__":
    hexy = hexapod()

    try:
        # zmq_url = "tcp://127.0.0.1:7001"
        # meshcat_vis = meshcat.Visualizer(zmq_url=zmq_url)
        viz = pin.visualize.MeshcatVisualizer(
            hexy.robot.model, hexy.robot.collision_model, hexy.robot.visual_model)
        viz.initViewer(open=True)
        # viz.initViewer(viewer=meshcat_vis, open=True)
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

    # des_pos = hexy.robot.framePlacement(
    #     hexy.robot.q0, 10).translation + np.array([0, 0, 0.025])
    # result = hexy.inverse_geometery(10, des_pos)
    # # hexy.robot.forwardKinematics(result)
    # print(
    #     f"Desired Pos = {des_pos} \nCurrent Pos = {hexy.robot.framePlacement(result, 10).translation}")
    # viz.display(result)

    hexy.generate_path_function()
    t_vals = np.linspace(0, 1, 5)
    # waypoints = [[round(hexy.x_t(t), 5), round(
    #     hexy.y_t(t), 5), round(hexy.z_t(t), 5)] for t in t_vals]
    waypoints = [[hexy.x_t(t), hexy.y_t(t), hexy.z_t(t)] for t in t_vals]
    # q = hexy.robot.q0
    # for wp in waypoints:
    #     q = hexy.inverse_geometery(q, 10, wp)
    #     viz.display(q)
    #     print(f"Current Pos = {hexy.robot.framePlacement(q, 10).translation}")
    # sleep(0.5)
    q_wps = [hexy.inverse_geometery(hexy.robot.q0, 10, wp) for wp in waypoints]
    t_init = 0
    t_goal = 1/10
    dt = 0.01

    for i in range(0, q_wps.__len__()-1):
        t = t_init
        while t < t_goal:
            viz.display(hexy.compute_trajectory(
                q_wps[i], q_wps[i+1], t_init, t_goal, t)[0])
            t = round((t + dt), 2)
    print("DEBUG POINT")
