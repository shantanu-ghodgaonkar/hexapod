# We import useful libraries
from time import sleep, time, strftime
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
from pathlib import Path
import sys
from scipy.optimize import minimize
import logging
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Tuple, NoReturn
import os
import jax

# We don't want to print every decimal!
np.set_printoptions(suppress=True, precision=4)


class hexapod:
    """
    A class representing a hexapod robot model using the Pinocchio library.

    This class provides methods to initialize the robot model, perform inverse kinematics,
    generate trajectories, and visualize the robot's movements.
    """

    def __init__(self, init_viz: bool = True, logging_level: int = logging.DEBUG) -> None:
        """
        Initialize the Hexapod robot model and visualization.

        Args:
            init_viz (bool): Whether to initialize the visualization. Default is True.
            logging_level (int): Logging level for the logger. Default is logging.DEBUG.
        """
        # Initialize the logger with the specified logging level
        self.init_logger(logging_level)
        # Set up the paths to the URDF model files
        self.pin_model_dir = str(
            Path('./URDF/Physics_Model_Edited').absolute())
        self.urdf_filename = (self.pin_model_dir +
                              '/Physics Model URDF (Edtd.).urdf')

        # Build the robot model using Pinocchio's RobotWrapper
        self.robot = pin.RobotWrapper.BuildFromURDF(
            self.urdf_filename, self.pin_model_dir, root_joint=pin.JointModelFreeFlyer())
        # Initialize the robot's kinematics and update geometry placements
        self.robot.forwardKinematics(pin.neutral(self.robot.model))
        self.robot.updateGeometryPlacements()
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        # Set up movement parameters
        self.direction_slope = 0.0
        self.HALF_STEP_SIZE_XY = 0.05 / 2  # Half of the step size in XY plane
        self.Y_CLEARANCE = 0.015
        self.Z_CLEARANCE = 0.015  # Step size in Z (vertical) direction
        # Retrieve frame IDs for the feet
        self.FOOT_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "foot" in frame.name]
        # Retrieve frame IDs for the shoulder joints
        self.SHOULDER_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "Revolute_joint_0" in frame.name]
        # Get the frame ID for the robot base
        self.BASE_FRAME_ID = self.robot.model.getFrameId("robot_base_inertial")
        self.cMb_se3 = self.robot.framePlacement(
            index=self.robot.model.getFrameId("CAM"), q=self.robot.q0).inverse()
        self.cMb_xyzquat = pin.SE3ToXYZQUAT(self.cMb_se3)
        # Set initial state and configuration
        self.state_flag = 'START'
        self.qc = self.robot.q0  # Initial joint configuration
        self.state_c = np.concatenate([
            self.robot.framePlacement(
                index=self.BASE_FRAME_ID, q=self.qc).translation,
            *[self.robot.framePlacement(index=i, q=self.qc).translation for i in self.FOOT_IDS]
        ]).reshape(-1, 1)
        # Initialize visualization if requested
        self.viz_flag = init_viz
        self.theta = np.pi / 4
        if self.viz_flag == True:
            self.init_viz()
        self.L_of = 0.058
        self.phi = np.pi/4
        # Weights for inverse geomtery in swing phase
        self.weights = np.array([2.0, 3.0, 5.0])  # Modify as needed
        for frame in self.robot.model.frames:
            # Log the frame name, type, ID, and position
            self.logger.info(
                f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")

        # self.cost_grad = jax.grad(self.cost_function)
        # Log successful initialization
        self.logger.info(
            f"Hexapod Object Initialised Successfully with init_viz = {self.viz_flag}, logging_level={logging_level}")

    def init_logger(self, logging_level: int) -> None:
        """
        Initialize the logger for the Hexapod class.

        Args:
            logging_level (int): Logging level (e.g., logging.DEBUG).
        """
        # Define the log directory
        log_dir = Path('logs').absolute()
        log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for log file
        timestamp = strftime("%Y%m%d_%H%M%S")
        log_file = f'{os.path.splitext(os.path.basename(__file__))[0]}_log_{timestamp}.log'
        log_path = log_dir / log_file

        # Create logger with the class name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)
        self.logger.propagate = False  # Prevent logging from propagating to root logger

        # Set up file handler for logging to file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Set up console handler for logging to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Define filters to exclude certain loggers
        excluded_loggers = ['some_imported_module', 'another_module']

        class ExcludeLoggersFilter(logging.Filter):
            def filter(self, record):
                return record.name not in excluded_loggers

        exclude_filter = ExcludeLoggersFilter()
        file_handler.addFilter(exclude_filter)
        console_handler.addFilter(exclude_filter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def init_viz(self) -> None:
        """
        Initialize the Meshcat visualizer for the robot.
        """
        try:
            # Initialize Meshcat visualizer
            self.viz = MeshcatVisualizer(
                self.robot.model, self.robot.collision_model, self.robot.visual_model)
            self.viz.initViewer(open=True)
        except ImportError as err:
            # Handle import error if Meshcat is not installed
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)

        # Load the robot model into the viewer
        self.viz.loadViewerModel()

        # Display frames and hide collisions
        self.viz.displayFrames(visibility=True)
        self.viz.displayCollisions(visibility=False)
        self.viz.display(self.qc)

    def print_all_frame_info(self) -> None:
        """
        Print information about all frames in the robot model.
        """
        for frame in self.robot.model.frames:
            # Print the frame name, type, ID, and position
            print(
                f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")

    def find_frame_info_by_name(self, name: str) -> int:
        """
        Find and print information about frames matching the given name.

        Args:
            name (str): Substring to match in frame names.

        Returns:
            int: Frame ID if found, otherwise None.
        """
        for frame in self.robot.model.frames:
            if name in frame.name:
                # Print the frame name, type, ID, and position
                print(
                    f"Frame found : Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")
                return self.robot.model.getFrameId(frame.name)

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the forward kinematics for the robot configuration q.

        Args:
            q (np.ndarray): Joint configuration vector.

        Returns:
            np.ndarray: Concatenated positions of the base and feet frames.
        """
        # Compute the forward kinematics and concatenate positions
        return np.concatenate([
            self.robot.framePlacement(
                index=self.BASE_FRAME_ID, q=q).translation,
            *[self.robot.framePlacement(index=i, q=q).translation for i in self.FOOT_IDS]
        ]).reshape(-1, 1)

    # def foot_pos_err(self, q: np.ndarray, FRAME_ID: int = 9, desired_pos: np.ndarray = np.zeros(3)) -> float:
    #     """
    #     Compute the weighted squared error between current and desired foot position.

    #     Args:
    #         q (np.ndarray): Joint configuration vector.
    #         FRAME_ID (int): Frame ID of the foot.
    #         desired_pos (np.ndarray): Desired position of the foot.

    #     Returns:
    #         float: Weighted squared error.
    #     """
    #     # self.robot.forwardKinematics(q)
    #     current_pos = self.robot.framePlacement(q, FRAME_ID).translation
    #     error = current_pos - desired_pos
    #     weighted_error = self.weights * error
    #     return np.dot(weighted_error, weighted_error)

    def cost_function(self, q: np.ndarray, desired_pos: np.ndarray = np.zeros((21, 1))) -> float:
        Q = np.eye(self.state_c.__len__())
        R = np.eye(self.robot.nq)
        # q = q.reshape(-1, 1)
        # if type(q) == jax._src.interpreters.ad.JVPTracer:
        #     q = q.primal
        current_pos = self.forward_kinematics(q)
        return (((current_pos - desired_pos).T @ Q @
                (current_pos - desired_pos))).item()
        # + (q.T @ R @ q)

    def get_jacobains(self, q: np.ndarray, desired_pos: np.ndarray = np.zeros((21, 1))) -> np.ndarray:
        return pin.computeJointJacobians(self.robot.model, self.robot.data, q)

    def cost_function_new(self, q: np.ndarray, desired_pos: np.ndarray = np.zeros((21, 1))) -> float:
        QR = np.eye(self.state_c.__len__() + self.robot.nq)
        # q = q.reshape(-1, 1)
        # if type(q) == jax._src.interpreters.ad.JVPTracer:
        #     q = q.primal
        current_pos = self.forward_kinematics(q)
        error = current_pos - desired_pos
        x = np.zeros((self.state_c.__len__() + self.robot.nq, 1))
        x[:self.state_c.__len__()] = error
        x[self.state_c.__len__():] = q.reshape(-1, 1)
        return (x.T @ QR @ x).item()

    def cost_function_new_grad(self, q: np.ndarray, desired_pos: np.ndarray = np.zeros((21, 1))) -> float:
        QR = np.eye(self.state_c.__len__() + self.robot.nq)
        # q = q.reshape(-1, 1)
        # if type(q) == jax._src.interpreters.ad.JVPTracer:
        #     q = q.primal
        current_pos = self.forward_kinematics(q)
        error = current_pos - desired_pos
        x = np.zeros((self.state_c.__len__() + self.robot.nq, 1))
        x[:self.state_c.__len__()] = error
        x[self.state_c.__len__():] = q.reshape(-1, 1)
        return 2 * QR @ x
        # + (q.T @ R @ q)

    # def cost_function_gradient(self, q: np.ndarray, desired_pos: np.ndarray = np.zeros((21, 1))) -> np.ndarray:
    #     Q = np.eye(self.state_c.__len__())
    #     R = np.eye(self.robot.nq)
    #     # q = q.reshape(-1, 1)
    #     current_pos = self.forward_kinematics(q)
    #     return (2 * Q @ (current_pos - desired_pos)).flatten()

    def equality_constraints(self, q: np.ndarray):
        desired = np.array([0, 0, 0, 0, 1])
        actual = q[2:7]
        return np.sum(np.abs(actual - desired))

    def feet_024_stop_constraints(self, q: np.ndarray):
        desired = self.state_c.flatten()
        state = self.forward_kinematics(q).flatten()
        actual_0 = state[3:6]
        actual_2 = state[9:12]
        actual_4 = state[15:18]
        return np.sum(np.abs(actual_0 - desired[3:6])) + np.sum(np.abs(actual_2 - desired[9:12])) + np.sum(np.abs(actual_4 - desired[15:18]))

    def feet_135_stop_constraints(self, q: np.ndarray):
        desired = self.state_c.flatten()
        state = self.forward_kinematics(q).flatten()
        actual_1 = state[6:9]
        actual_3 = state[12:15]
        actual_5 = state[18:21]
        return np.sum(np.abs(actual_1 - desired[6:9])) + np.sum(np.abs(actual_3 - desired[12:15])) + np.sum(np.abs(actual_5 - desired[18:21]))

    def inverse_geometery(self, q: np.ndarray, FRAME_ID: int = 9, desired_pos: np.ndarray = np.zeros(3)) -> np.ndarray:
        """
        Perform inverse kinematics to find joint configuration that places a frame at the desired position.

        Args:
            q (np.ndarray): Initial guess for the joint configuration.
            FRAME_ID (int): Frame ID for which to compute inverse kinematics.
            desired_pos (np.ndarray): Desired position for the frame.

        Returns:
            np.ndarray: Joint configuration that minimizes the positional error.
        """
        # Set bounds for optimization
        # bounds = [((self.qc[i], self.qc[i])) for i in range(self.robot.nq)]
        # bounds[int(7 + (3 * (np.floor(FRAME_ID / 8) - 1))): int(7 + (3 * (np.floor(FRAME_ID / 8) - 1)))+3] = \
        #     [(-(85 * np.pi / 180), (85 * np.pi / 180)), (-(45 * np.pi / 180),
        #                                                  (45 * np.pi / 180)), (-(45 * np.pi / 180), (45 * np.pi / 180))]

        # Default to fixed joint values
        # start = time()
        joint_bounds = np.full((self.robot.nq, 2), self.qc[:, None])
        joint_indices = np.array(
            [7 + (3 * ((FRAME_ID // 8) - 1)) + i for i in range(3)])  # Indices for this leg
        joint_bounds[joint_indices, 0] = [-85 * np.pi /
                                          180, -45 * np.pi / 180, -45 * np.pi / 180]
        joint_bounds[joint_indices, 1] = [
            85 * np.pi / 180, 45 * np.pi / 180, 45 * np.pi / 180]
        bounds = list(map(tuple, joint_bounds))

        # Perform minimization to find joint configuration minimizing foot position error
        res = minimize(
            self.foot_pos_err, q, args=(
                FRAME_ID, desired_pos),
            bounds=bounds, tol=1e-8, method='L-BFGS-B', options={'disp': False})
        # Return the optimized joint configuration
        # print(f"Optimised in {time()-start}")
        return res.x

    def north_vector(self) -> np.ndarray:
        """
        Compute the unit vector pointing in the 'north' direction based on the robot's configuration.

        Returns:
            np.ndarray: Unit vector in the 'north' direction.
        """
        # Get positions of the first two shoulder joints
        p1 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[0]).translation
        p2 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[1]).translation
        # Compute vector orthogonal to the line between p1 and p2
        v = np.array([(p2 - p1)[1], -(p2 - p1)[0]])
        # Normalize the vector to create a unit vector
        v = v/(np.sqrt((v[0]**2) + (v[1]**2)))
        # Visualize the direction vector if visualization is enabled
        if self.viz_flag:
            self.viz.viewer['direction_vector'].set_object(
                g.Line(g.PointsGeometry(np.array(
                    ([0, 0, 0], np.hstack((v, [0.0])))).T
                ), g.MeshBasicMaterial(color=0xffff00)))
        return v

    def south_vector(self) -> np.ndarray:
        """
        Compute the unit vector pointing in the 'south' direction based on the robot's configuration.

        Returns:
            np.ndarray: Unit vector in the 'south' direction.
        """
        # Get positions of the shoulder joints
        p1 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[3]).translation
        p2 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[4]).translation
        # Compute vector orthogonal to the line between p1 and p2
        v = np.array([(p2 - p1)[1], -(p2 - p1)[0]])
        # Normalize the vector to create a unit vector
        v = v/(np.sqrt((v[0]**2) + (v[1]**2)))
        # Visualize the direction vector if visualization is enabled
        if self.viz_flag:
            self.viz.viewer['direction_vector'].set_object(
                g.Line(g.PointsGeometry(np.array(
                    ([0, 0, 0], np.hstack((v, [0.0])))).T
                ), g.MeshBasicMaterial(color=0xffff00)))
        return v

    def east_vector(self) -> np.ndarray:
        """
        Compute the unit vector pointing in the 'east' direction based on the robot's configuration.

        Returns:
            np.ndarray: Unit vector in the 'east' direction.
        """
        # Get positions of the shoulder joints
        p1 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[1]).translation
        p2 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[3]).translation
        # Compute vector orthogonal to the line between p1 and p2
        v = np.array([(p2 - p1)[1], -(p2 - p1)[0]])
        # Normalize the vector to create a unit vector
        v = v/(np.sqrt((v[0]**2) + (v[1]**2)))
        # Visualize the direction vector if visualization is enabled
        if self.viz_flag:
            self.viz.viewer['direction_vector'].set_object(
                g.Line(g.PointsGeometry(np.array(
                    ([0, 0, 0], np.hstack((v, [0.0])))).T
                ), g.MeshBasicMaterial(color=0xffff00)))
        return v

    def west_vector(self) -> np.ndarray:
        """
        Compute the unit vector pointing in the 'west' direction based on the robot's configuration.

        Returns:
            np.ndarray: Unit vector in the 'west' direction.
        """
        # Get positions of the shoulder joints
        p1 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[4]).translation
        p2 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[0]).translation
        # Compute vector orthogonal to the line between p1 and p2
        v = np.array([(p2 - p1)[1], -(p2 - p1)[0]])
        # Normalize the vector to create a unit vector
        v = v/(np.sqrt((v[0]**2) + (v[1]**2)))
        # Visualize the direction vector if visualization is enabled
        if self.viz_flag:
            self.viz.viewer['direction_vector'].set_object(
                g.Line(g.PointsGeometry(np.array(
                    ([0, 0, 0], np.hstack((v, [0.0])))).T
                ), g.MeshBasicMaterial(color=0xffff00)))
        return v

    def north_east_vector(self) -> np.ndarray:
        """
        Compute the unit vector pointing in the 'northeast' direction based on the robot's configuration.

        Returns:
            np.ndarray: Unit vector in the 'northeast' direction.
        """
        # Get positions of the shoulder joints
        p1 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[1]).translation
        p2 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[2]).translation
        # Compute vector orthogonal to the line between p1 and p2
        v = np.array([(p2 - p1)[1], -(p2 - p1)[0]])
        # Normalize the vector to create a unit vector
        v = v/(np.sqrt((v[0]**2) + (v[1]**2)))
        # Visualize the direction vector if visualization is enabled
        if self.viz_flag:
            self.viz.viewer['direction_vector'].set_object(
                g.Line(g.PointsGeometry(np.array(
                    ([0, 0, 0], np.hstack((v, [0.0])))).T
                ), g.MeshBasicMaterial(color=0xffff00)))
        return v

    def north_west_vector(self) -> np.ndarray:
        """
        Compute the unit vector pointing in the 'northwest' direction based on the robot's configuration.

        Returns:
            np.ndarray: Unit vector in the 'northwest' direction.
        """
        # Get positions of the shoulder joints
        p1 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[5]).translation
        p2 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[0]).translation
        # Compute vector orthogonal to the line between p1 and p2
        v = np.array([(p2 - p1)[1], -(p2 - p1)[0]])
        # Normalize the vector to create a unit vector
        v = v/(np.sqrt((v[0]**2) + (v[1]**2)))
        # Visualize the direction vector if visualization is enabled
        if self.viz_flag:
            self.viz.viewer['direction_vector'].set_object(
                g.Line(g.PointsGeometry(np.array(
                    ([0, 0, 0], np.hstack((v, [0.0])))).T
                ), g.MeshBasicMaterial(color=0xffff00)))
        return v

    def south_east_vector(self) -> np.ndarray:
        """
        Compute the unit vector pointing in the 'southeast' direction based on the robot's configuration.

        Returns:
            np.ndarray: Unit vector in the 'southeast' direction.
        """
        # Get positions of the shoulder joints
        p1 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[2]).translation
        p2 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[3]).translation
        # Compute vector orthogonal to the line between p1 and p2
        v = np.array([(p2 - p1)[1], -(p2 - p1)[0]])
        # Normalize the vector to create a unit vector
        v = v/(np.sqrt((v[0]**2) + (v[1]**2)))
        # Visualize the direction vector if visualization is enabled
        if self.viz_flag:
            self.viz.viewer['direction_vector'].set_object(
                g.Line(g.PointsGeometry(np.array(
                    ([0, 0, 0], np.hstack((v, [0.0])))).T
                ), g.MeshBasicMaterial(color=0xffff00)))
        return v

    def south_west_vector(self) -> np.ndarray:
        """
        Compute the unit vector pointing in the 'southwest' direction based on the robot's configuration.

        Returns:
            np.ndarray: Unit vector in the 'southwest' direction.
        """
        # Get positions of the shoulder joints
        p1 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[4]).translation
        p2 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[5]).translation
        # Compute vector orthogonal to the line between p1 and p2
        v = np.array([(p2 - p1)[1], -(p2 - p1)[0]])
        # Normalize the vector to create a unit vector
        v = v/(np.sqrt((v[0]**2) + (v[1]**2)))
        # Visualize the direction vector if visualization is enabled
        if self.viz_flag:
            self.viz.viewer['direction_vector'].set_object(
                g.Line(g.PointsGeometry(np.array(
                    ([0, 0, 0], np.hstack((v, [0.0])))).T
                ), g.MeshBasicMaterial(color=0xffff00)))
        return v

    def default_vector(self) -> NoReturn:
        """
        Raise an error for invalid direction inputs.

        Raises:
            KeyError: If the direction is not valid.
        """
        raise KeyError(
            "NOT A VAILD DIRECTION. CHOOSE ONE OF {N, S, E, W, NE, NW, SE, SW} ONLY")

    def generate_direction_vector(self, DIR: str = 'N') -> np.ndarray:
        """
        Generate a unit vector corresponding to the specified direction.

        Args:
            DIR (str): Direction string, one of {'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'}.

        Returns:
            np.ndarray: Unit vector in the specified direction.
        """
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

    def init_foot_trajectory_functions(self, step_size_xy_mult: float, DIR: str = 'N', FOOT_ID: int = 10) -> None:
        """
        Initialize the foot trajectory functions for x, y, and z over time t.

        Args:
            step_size_xy_mult (float): Multiplier for the step size in the XY plane.
            DIR (str): Direction of movement ('N', 'S', etc.).
            FOOT_ID (int): Frame ID of the foot.
        """
        # Generate direction vector based on specified direction
        v = self.generate_direction_vector(DIR=DIR)
        # Get initial position of the foot in XY plane
        p1 = self.robot.framePlacement(self.qc, FOOT_ID).translation
        # Compute the target position p2 in XY plane
        p2 = p1[0:2] + (self.HALF_STEP_SIZE_XY * step_size_xy_mult * v)
        # Define trajectory functions for x(t) and y(t) as linear interpolation between p1 and p2
        p2 = np.append(p2, p1[2])

        # Apex points (midway)
        y_apex = max(p1[1], p2[1]) + (self.Y_CLEARANCE *
                                      np.sign(max(p1[1], p2[1])))
        z_apex = max(p1[2], p2[2]) + (self.Z_CLEARANCE *
                                      np.sign(max(p1[2], p2[2])))

        # Solve for parabolic coefficients for y(t) = a_y * t^2 + b_y * t + c_y
        A_y = np.array([
            [0**2, 0, 1],  # y(0) = y0
            [0.5**2, 0.5, 1],  # y(0.5) = y_apex
            [1**2, 1, 1]   # y(1) = y1
        ])
        b_y = np.array([p1[1], y_apex, p2[1]])
        a_y, b_y, c_y = np.linalg.solve(A_y, b_y)

        # Solve for parabolic coefficients for z(t) = a_z * t^2 + b_z * t + c_z
        A_z = np.array([
            [0**2, 0, 1],  # z(0) = z0
            [0.5**2, 0.5, 1],  # z(0.5) = z_apex
            [1**2, 1, 1]   # z(1) = z1
        ])
        b_z = np.array([p1[2], z_apex, p2[2]])
        a_z, b_z, c_z = np.linalg.solve(A_z, b_z)

        # Define trajectory functions for x, y, and z
        self.x_t = lambda t: (1 - t) * p1[0] + t * p2[0]
        self.y_t = lambda t: a_y * t**2 + b_y * t + c_y
        self.z_t = lambda t: a_z * t**2 + b_z * t + c_z

    def init_body_trajectory_functions(self, step_size_xy_mult: float, DIR: str = 'N') -> None:
        """
        Initialize the body trajectory functions for x and y over time t.

        Args:
            step_size_xy_mult (float): Multiplier for step size in XY plane.
            DIR (str): Direction of movement ('N', 'S', etc.).
        """
        # Generate direction vector based on specified direction
        v = self.generate_direction_vector(DIR=DIR)
        # Get initial position of the base in XY plane
        p1 = self.robot.framePlacement(self.qc, self.BASE_FRAME_ID).translation
        # Compute the target position p2 in XY plane
        p2 = p1[0:2] + (self.HALF_STEP_SIZE_XY * step_size_xy_mult * v)
        # Define trajectory functions for x(t) and y(t) as linear interpolation between p1 and p2
        p2 = np.append(p2, p1[2])
        self.x_t = lambda t: ((1 - t) * p1[0]) + (t * p2[0])
        self.y_t = lambda t:  ((1 - t) * p1[1]) + (t * p2[1])

    def generate_joint_waypoints(self, step_size_xy_mult: float, WAYPOINTS: int = 5, DIR: str = 'N', FOOT_ID: int = 10) -> List[np.ndarray]:
        """
        Generate joint waypoints for a foot trajectory.

        Args:
            step_size_xy_mult (float): Multiplier for step size in XY plane.
            WAYPOINTS (int): Number of waypoints in the trajectory.
            DIR (str): Direction of movement ('N', 'S', etc.).
            FOOT_ID (int): Frame ID of the foot.

        Returns:
            list: List of joint configurations along the trajectory.
        """
        # Initialize foot trajectory functions
        self.init_foot_trajectory_functions(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, FOOT_ID=FOOT_ID)
        # Create time steps from 0 to 1
        s = np.linspace(0, 1, WAYPOINTS)
        # Generate waypoints by evaluating trajectory functions at each time step
        waypoints = [[round(self.x_t(t), 5), round(
            self.y_t(t), 5), round(self.z_t(t), 5)] for t in s]
        # Convert waypoints to array for visualization
        points = np.array([waypoints[0], waypoints[-1]]).T

        # Visualize the foot trajectory if visualization is enabled
        # if self.viz_flag:
        #     self.viz.viewer[('Foot_ID_' + str(FOOT_ID) + '_trajectory')].set_object(
        #         g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xffff00)))

        # Perform inverse kinematics to get joint configurations for each waypoint
        return [self.inverse_geometery(q=self.qc, FRAME_ID=FOOT_ID, desired_pos=wp)
                for wp in waypoints]

    def generate_waypoints(self, step_size_xy_mult: float, WAYPOINTS: int = 5, DIR: str = 'N', leg_set: str = '024'):
        start = None
        if (leg_set == '024'):
            start = 0
        elif (leg_set == '135'):
            start = 1
        else:
            raise ValueError(
                f'Expected value for leg_set is a string "024" or "135". Got {leg_set} instead')

        waypoints = self.state_c @ np.ones((1, WAYPOINTS))

        # Initialize body trajectory functions
        self.init_body_trajectory_functions(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR)
        s = np.linspace(0, 1, WAYPOINTS)
        # Generate waypoints by evaluating trajectory functions at each time step
        wp = np.array([[round(self.x_t(t), 5), round(
            self.y_t(t), 5), self.state_c[2][0]] for t in s]).T

        waypoints[0:3, :] = wp
        for foot in self.FOOT_IDS[start::2]:
            # Initialize foot trajectory functions
            self.init_foot_trajectory_functions(
                step_size_xy_mult=step_size_xy_mult, DIR=DIR, FOOT_ID=foot)
            # Create time steps from 0 to 1
            s = np.linspace(0, 1, WAYPOINTS)
            # Generate waypoints by evaluating trajectory functions at each time step
            wp = np.array([[round(self.x_t(t), 5), round(
                self.y_t(t), 5), round(self.z_t(t), 5)] for t in s]).T
            idx = (3 * (foot // 8 - 1) + 3)
            waypoints[idx:idx+3, :] = wp

        return waypoints

    def compute_trajectory_pva(self, position_init: np.ndarray, position_goal: np.ndarray, t_init: float, t_goal: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Time parameterization of the trajectory with acceleration and velocity constraints.

        Args:
            position_init (numpy.ndarray): Initial position in Joint state form.
            position_goal (numpy.ndarray): Goal position in Joint state form.
            t_init (float): Initial time.
            t_goal (float): Goal time.
            t (float): Current time.

        Returns:
            tuple: Desired position, velocity, and acceleration at time t.
        """
        t_tot = t_goal - t_init
        # Compute normalized time variable between 0 and 1
        tau = (t - t_init) / t_tot
        position_diff = position_goal - position_init
        # Compute position using a 5th-degree polynomial (quintic trajectory)
        self.desired_position = position_init + (
            ((10 * tau**3) - (15 * tau**4) + (6 * tau**5)) * position_diff)
        # Compute velocity
        self.desired_velocity = (
            ((30 * tau**2) - (60 * tau**3) + (30 * tau**4)) * position_diff / t_tot)
        # Compute acceleration
        self.desired_acceleration = (
            ((60 * tau) - (180 * tau**2) + (120 * tau**3)) * position_diff / (t_tot**2))

        return self.desired_position, self.desired_velocity, self.desired_acceleration

    def compute_trajectory_pv(self, position_init: np.ndarray, position_goal: np.ndarray, t_init: float, t_goal: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the desired trajectory, velocity, and acceleration at time t
        using third-degree (cubic) polynomial equations.

        Args:
            position_init (numpy.ndarray): Initial position in Joint state form.
            position_goal (numpy.ndarray): Goal position in Joint state form.
            t_init (float): Initial time.
            t_goal (float): Goal time.
            t (float): Current time.

        Returns:
            tuple: Desired position, velocity, and acceleration at time t.
        """
        T = t_goal - t_init
        if T <= 0:
            raise ValueError("t_goal must be greater than t_init.")

        tau = (t - t_init) / T
        # Clamp tau to the range [0, 1] to handle times outside the trajectory duration
        tau = np.clip(tau, 0.0, 1.0)

        theta_diff = position_goal - position_init

        # Compute desired position using cubic polynomial
        self.desired_position = position_init + \
            (3 * tau**2 - 2 * tau**3) * theta_diff

        # Compute desired velocity
        self.desired_velocity = (6 * tau - 6 * tau**2) * theta_diff / T

        # Compute desired acceleration
        self.desired_acceleration = (6 - 12 * tau) * theta_diff / (T**2)

        return self.desired_position, self.desired_velocity, self.desired_acceleration

    def compute_trajectory_p(self, position_init: np.ndarray, position_goal: np.ndarray, t_init: float, t_goal: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the desired position, velocity, and acceleration at time t
        using linear time parametrization without using a normalized time variable τ.

        Args:
            position_init (numpy.ndarray): Initial position in joint state form.
            position_goal (numpy.ndarray): Goal position in joint state form.
            t_init (float): Initial time.
            t_goal (float): Goal time.
            t (float): Current time.

        Returns:
            tuple: Desired position, velocity, and acceleration at time t.
        """
        T = t_goal - t_init
        if T <= 0:
            raise ValueError("t_goal must be greater than t_init.")

        delta_t = t - t_init
        theta_diff = position_goal - position_init

        # Clamp delta_t to the range [0, T] to handle times outside the trajectory duration
        delta_t_clamped = np.clip(delta_t, 0.0, T)

        # Compute the fraction of time elapsed
        fraction = delta_t_clamped / T

        # Compute desired position using linear interpolation
        self.desired_position = position_init + fraction * theta_diff

        # Compute desired velocity (constant)
        self.desired_velocity = theta_diff / T

        # Compute desired acceleration (zero)
        self.desired_acceleration = np.zeros_like(position_init)

        return self.desired_position, self.desired_velocity, self.desired_acceleration

    def generate_leg_joint_trajectory(self, step_size_xy_mult: float, DIR: str = 'N', LEG: int = 0, WAYPOINTS: int = 5, t_init: float = 0, t_goal: float = 0.1, dt: float = 0.01) -> np.ndarray:
        """
        Generate a joint trajectory for a specific leg.

        Args:
            step_size_xy_mult (float): Multiplier for step size in XY plane.
            DIR (str): Direction of movement ('N', 'S', etc.).
            LEG (int): Index of the leg (0 to 5).
            WAYPOINTS (int): Number of waypoints in the trajectory.
            t_init (float): Initial time.
            t_goal (float): Goal time.
            dt (float): Time step for trajectory generation.

        Returns:
            numpy.ndarray: Array of joint configurations along the trajectory.
        """
        # Generate joint waypoints for the specified leg
        q_wps = self.generate_joint_waypoints(step_size_xy_mult,
                                              DIR=DIR, FOOT_ID=self.FOOT_IDS[LEG], WAYPOINTS=WAYPOINTS)
        q_traj = []
        # Create a mask to apply joint configurations to the specific leg
        mask = np.concatenate((np.zeros(6), [1], np.zeros(
            LEG*3), [1, 1, 1], np.zeros((5-LEG)*3)))
        for i in range(0, q_wps.__len__()-1):
            t = t_init
            while round(t, 3) <= t_goal:
                # Compute trajectory using linear interpolation
                q_t = self.compute_trajectory_p(
                    q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
                q_traj.append(np.multiply(q_t, mask))
                t = (t + dt)
        # Remove the initial configuration
        return np.vstack(q_traj)

    def get_foot_positions(self, q: np.ndarray) -> List[np.ndarray]:
        """
        Get the positions of all feet for a given joint configuration.

        Args:
            q (numpy.ndarray): Joint configuration vector.

        Returns:
            list: List of foot positions.
        """
        return [self.robot.framePlacement(q, foot_id).translation for foot_id in self.FOOT_IDS]

    def feet_error(self, q_joints: np.ndarray, desired_base_pose: np.ndarray) -> float:
        """
        Compute the sum of squared errors between current and desired foot positions.

        Args:
            q_joints (numpy.ndarray): Joint angles excluding base joints.
            desired_base_pose (numpy.ndarray): Desired base pose.

        Returns:
            float: Sum of squared positional errors for all feet.
        """
        # Record initial foot positions
        initial_foot_positions = self.get_foot_positions(self.qc)
        q_full = np.concatenate([desired_base_pose, q_joints])
        self.robot.forwardKinematics(q_full)
        # error = 0
        # for foot_id, desired_pos in zip(self.FOOT_IDS, initial_foot_positions):
        #     current_pos = self.robot.framePlacement(
        #         q_full, foot_id).translation
        #     error += np.linalg.norm(current_pos - desired_pos)**2

        current_positions = np.array([self.robot.framePlacement(
            q_full, fid).translation for fid in self.FOOT_IDS])
        desired_positions = np.array(initial_foot_positions)
        error = np.sum(np.linalg.norm(current_positions -
                       desired_positions, axis=1) ** 2)

        return error

    def body_inverse_geometry(self, q: np.ndarray, desired_base_pos: np.ndarray) -> np.ndarray:
        """
        Compute inverse kinematics for the robot's body to maintain foot positions.

        Args:
            q (numpy.ndarray): Current joint configuration.
            desired_base_pos (numpy.ndarray): Desired base position.

        Returns:
            numpy.ndarray: Joint configuration that maintains foot positions.
        """
        # Joint angle bounds (exclude base joints)
        bounds = [(-(48 * np.pi / 180), (48 * np.pi / 180))] * \
            (self.robot.nq - 7)

        # Initial joint angles
        q_joints_init = q[7:].copy()

        res = minimize(
            self.feet_error,
            q_joints_init, args=(desired_base_pos),
            bounds=bounds,
            method='L-BFGS-B', options={'disp': False},
            tol=1e-8
        )

        return np.concatenate([desired_base_pos, res.x])

    def generate_body_path_waypoints(self, step_size_xy_mult: float = 1, WAYPOINTS: int = 5, DIR: str = 'N') -> List[np.ndarray]:
        """
        Generate waypoints for the robot body's path.

        Args:
            step_size_xy_mult (float): Multiplier for step size in XY plane.
            WAYPOINTS (int): Number of waypoints in the trajectory.
            DIR (str): Direction of movement ('N', 'S', etc.).

        Returns:
            list: List of joint configurations along the body's path.
        """
        # Initialize body trajectory functions
        self.init_body_trajectory_functions(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR)
        s = np.linspace(0, 1, WAYPOINTS)
        # Generate waypoints by evaluating trajectory functions at each time step
        waypoints = [np.concatenate(([round(self.x_t(t), 5), round(
            self.y_t(t), 5)], self.qc[2:7].copy())) for t in s]

        points = np.array(waypoints)[:, 0:3].T
        # Visualize the base trajectory if visualization is enabled
        # if self.viz_flag:
        #     self.viz.viewer[('Base_trajectory')].set_object(
        #         g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xffff00)))
        # Perform inverse kinematics to get joint configurations for each waypoint
        return [self.body_inverse_geometry(self.qc, wp)
                for wp in waypoints]

    def generate_body_joint_trajectory(self, step_size_xy_mult: float, DIR: str = 'N', WAYPOINTS: int = 5, t_init: float = 0, t_goal: float = 0.1, dt: float = 0.01) -> np.ndarray:
        """
        Generate a joint trajectory for the robot's body.

        Args:
            step_size_xy_mult (float): Multiplier for step size in XY plane.
            DIR (str): Direction of movement ('N', 'S', etc.).
            WAYPOINTS (int): Number of waypoints in the trajectory.
            t_init (float): Initial time.
            t_goal (float): Goal time.
            dt (float): Time step for trajectory generation.

        Returns:
            numpy.ndarray: Array of joint configurations along the trajectory.
        """
        # Generate waypoints for the body's path
        q_wps = self.generate_body_path_waypoints(step_size_xy_mult,
                                                  DIR=DIR, WAYPOINTS=WAYPOINTS)
        q_traj = []
        for i in range(0, q_wps.__len__()-1):
            t = t_init
            while round(t, 3) <= t_goal:
                # Compute trajectory using linear interpolation
                q_t = self.compute_trajectory_p(
                    q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
                q_traj.append(q_t)
                t = (t + dt)
        # Remove the initial configuration
        return np.vstack(q_traj)

    def compute_gait(self, v: float = 0.5, WAYPOINTS: int = 5, STEP_CNT: int = 3, DIR: str = 'N') -> np.ndarray:
        """
        Compute the gait trajectory for the robot.

        Args:
            v (float): Velocity in m/s.
            WAYPOINTS (int): Number of waypoints per step.
            STEP_CNT (int): Number of steps.
            DIR (str): Direction of movement.

        Returns:
            np.ndarray: The gait trajectory as a numpy array of joint configurations.
        """
        step_size_xy_mult = 1
        t_goal = self.HALF_STEP_SIZE_XY / v
        start_time = time()
        # Generate trajectories for legs 0, 2, and 4
        start = time()
        leg0_traj = self.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=0, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
        leg2_traj = self.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=2, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
        leg4_traj = self.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=4, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
        # Generate body trajectory
        body_traj = self.generate_body_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
        # Combine the trajectories into a single trajectory q
        q = np.hstack((body_traj[:, 0:7], leg0_traj[:, 7:10], body_traj[:, 10:13],
                       leg2_traj[:, 13:16], body_traj[:, 16:19], leg4_traj[:, 19:22], body_traj[:, 22:25]))
        # Update the current configuration
        self.qc = q[-1]
        step_size_xy_mult = 2
        for i in range(0, STEP_CNT):
            # Generate trajectories for legs 1, 3, and 5
            leg1_traj = self.generate_leg_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=1, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
            leg3_traj = self.generate_leg_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=3, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
            leg5_traj = self.generate_leg_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=5, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
            # Generate body trajectory
            body_traj = self.generate_body_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR=DIR, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
            # Append trajectories to q
            q = np.vstack((q,
                           np.hstack((body_traj[:, 0:7], body_traj[:, 7:10],
                                      leg1_traj[:, 10:13], body_traj[:, 13:16],
                                      leg3_traj[:, 16:19], body_traj[:, 19:22], leg5_traj[:, 22:25]))
                           ))
            # Update the current configuration
            self.qc = q[-1]
            # Generate trajectories for legs 0, 2, and 4 again
            leg0_traj = self.generate_leg_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=0, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
            leg2_traj = self.generate_leg_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=2, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
            leg4_traj = self.generate_leg_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=4, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
            # Generate body trajectory
            body_traj = self.generate_body_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR=DIR, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
            # Append trajectories to q
            q = np.vstack((q,
                           np.hstack((body_traj[:, 0:7], leg0_traj[:, 7:10],
                                      body_traj[:, 10:13], leg2_traj[:, 13:16],
                                      body_traj[:, 16:19], leg4_traj[:, 19:22], body_traj[:, 22:25]))
                           ))
            # Update the current configuration
            self.qc = q[-1]

        step_size_xy_mult = 1
        # Generate trajectories for legs 1, 3, and 5
        leg1_traj = self.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=1, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
        leg3_traj = self.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=3, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
        leg5_traj = self.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=5, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
        # Generate body trajectory
        body_traj = self.generate_body_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
        # Append trajectories to q
        q = np.vstack((q,
                       np.hstack((body_traj[:, 0:7], body_traj[:, 7:10],
                                  leg1_traj[:, 10:13], body_traj[:, 13:16],
                                  leg3_traj[:, 16:19], body_traj[:, 19:22], leg5_traj[:, 22:25]))
                       ))
        # Update the current configuration
        q = np.vstack((q, np.hstack((q[-1, :7], np.zeros(self.robot.nq - 7)))))
        self.qc = q[-1]
        # Log the time taken to compute the steps
        self.logger.debug(
            f'Time taken for computing {(STEP_CNT*2)+2} steps = {time()-start_time}')

        return q

    def plot_trajctory(self, state: np.ndarray, title: str) -> None:
        """
        Plot the trajectory of the robot's base and legs over time.

        Args:
            state (np.ndarray): Array of positions over time.
            title (str): Title of the plot.
        """
        x_figsize = 15
        y_figsize = 30
        # Create a time array based on the number of columns in x1 or x2
        time = np.arange(state.shape[0])

        # Create a single figure with a 7-row, 3-column layout
        fig, axs = plt.subplots(7, 3, figsize=(x_figsize, y_figsize))
        fig.suptitle(title)

        # Plot base positions
        axs[0, 0].scatter(time, state[:, 0], label='x body', color='blue')
        axs[0, 0].set_xlabel('time')
        axs[0, 0].set_ylabel('x body')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].scatter(time, state[:, 1], label='y body', color='blue')
        axs[0, 1].set_xlabel('time')
        axs[0, 1].set_ylabel('y body')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[0, 2].scatter(time, state[:, 2], label='z body', color='blue')
        axs[0, 2].set_xlabel('time')
        axs[0, 2].set_ylabel('z body')
        axs[0, 2].legend()
        axs[0, 2].grid(True)

        # Plot leg positions
        for leg in range(6):
            row = leg + 1
            axs[row, 0].scatter(time, state[:, 3 + leg * 3],
                                label=f'x leg {leg}', color='blue')
            axs[row, 0].set_xlabel('time')
            axs[row, 0].set_ylabel(f'x leg {leg}')
            axs[row, 0].legend()
            axs[row, 0].grid(True)

            axs[row, 1].scatter(time, state[:, 4 + leg * 3],
                                label=f'y leg {leg}', color='blue')
            axs[row, 1].set_xlabel('time')
            axs[row, 1].set_ylabel(f'y leg {leg}')
            axs[row, 1].legend()
            axs[row, 1].grid(True)

            axs[row, 2].scatter(time, state[:, 5 + leg * 3],
                                label=f'z leg {leg}', color='blue')
            axs[row, 2].set_xlabel('time')
            axs[row, 2].set_ylabel(f'z leg {leg}')
            axs[row, 2].legend()
            axs[row, 2].grid(True)

        plt.show()


if __name__ == "__main__":
    # Create a hexapod instance with visualization and debug logging

    hexy = hexapod(init_viz=False, logging_level=logging.DEBUG)
    sleep(2)
    # Set parameters for movement
    v = 0.5  # Velocity in m/s
    # start_time = time()
    WAYPOINTS = 20
    # DIR = 'N'
    step_size_mult = 1
    start = time()
    wp = hexy.generate_waypoints(
        WAYPOINTS=WAYPOINTS, step_size_xy_mult=step_size_mult)
    q = np.zeros((WAYPOINTS, hexy.robot.nq))
    for i in range(WAYPOINTS):
        q[i, :] = minimize(fun=hexy.cost_function, x0=hexy.qc, args=wp[:, i].reshape(
            -1, 1), constraints=[{'type': 'eq', 'fun': hexy.equality_constraints},
                                 #  {'type': 'eq', 'fun': hexy.feet_135_stop_constraints}
                                 ], method='SLSQP', options={'disp': True}, tol=1e-7,
            # jac=hexy.cost_grad
            # jac=hexy.cost_function_new_grad
        ).x
        # hexy.qc = minimize(fun=hexy.cost_function, x0=hexy.qc, args=wp[:, 0].reshape(
        #     -1, 1), constraints=[{'type': 'eq', 'fun': hexy.equality_constraints},
        #                          #  {'type': 'eq', 'fun': hexy.feet_135_stop_constraints}
        #                          ], method='SLSQP', options={'disp': True}, tol=1e-7,
        #     # jac=hexy.cost_grad
        #     # jac=hexy.cost_function_new_grad
        # ).x
        hexy.qc = q[i, :]
        hexy.state_c = hexy.forward_kinematics(hexy.qc)
        # hexy.viz.display(hexy.qc)
    print(f'Computation done in {time()-start} seconds')
    states = np.array([hexy.forward_kinematics(q_i) for q_i in q])
    hexy.plot_trajctory(state=states, title='v2.5.1')
    # sleep(3)
    # STEP_CNT = 1
    # # Compute the gait trajectory
    # q = hexy.compute_gait(v=v, WAYPOINTS=WAYPOINTS, STEP_CNT=STEP_CNT, DIR=DIR)
    # q_traj = []
    # q = np.round(q, )
    # t_goal = 0.2
    # dt = 0.01
    # q_traj = np.zeros((int(((t_goal/dt)*(WAYPOINTS-1)) + 1), q.shape[1]))
    # q_traj_iter = 0
    # for i in range(q.shape[0] - 1):
    #     t = 0
    #     while t < t_goal:
    #         q_traj[q_traj_iter, :] = hexy.compute_trajectory_p(
    #             position_init=q[i, :], position_goal=q[i+1, :], t_init=0, t_goal=t_goal, t=t)[0]
    #         t += dt
    #         q_traj_iter += 1

    # q_traj[-1, :] = q[-1, :]

    # states = np.array([hexy.forward_kinematics(q_i) for q_i in q_traj])
    # hexy.plot_trajctory(state=states, title='v2.5.1')

    # # Save the gait angles to a file
    # gait_angles_file_path = Path(
    #     f'gait_angles/gait_angles_DIR_{DIR}_WP{WAYPOINTS}_S{STEP_CNT}_{strftime("%Y%m%d_%H%M%S")}.npy')
    # gait_angles_file_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(gait_angles_file_path, q)
    # # Play the trajectory in the visualizer if enabled
    # if hexy.viz_flag:
    #     hexy.viz.play(q)

    # sleep(1)
    # STEP_CNT = 10
    # Compute the gait trajectory for more steps
    # q = hexy.compute_gait(v=v, WAYPOINTS=WAYPOINTS, STEP_CNT=STEP_CNT, DIR=DIR)
    # if hexy.viz_flag:
    #     hexy.viz.play(q)

    # # Save the gait angles to a file
    # gait_angles_file_path = Path(
    #     f'gait_angles/gait_angles_DIR_{DIR}_WP{WAYPOINTS}_S{STEP_CNT}_{strftime("%Y%m%d_%H%M%S")}.npy')
    # gait_angles_file_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(gait_angles_file_path, q)

    # for DIR in ['N', 'S', 'E', 'W', 'NE', 'SE', 'NW', 'SW']:
    #     step_size_xy_mult = 1
    #     t_goal = hexy.HALF_STEP_SIZE_XY / v
    #     # Generate trajectories for legs 0, 2, and 4
    #     leg0_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=0, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     leg2_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=2, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     leg4_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=4, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     # Generate body trajectory
    #     body_traj = hexy.generate_body_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     # Combine the trajectories into a single trajectory q
    #     q = np.hstack((body_traj[:, 0:7], leg0_traj[:, 7:10], body_traj[:, 10:13],
    #                    leg2_traj[:, 13:16], body_traj[:, 16:19], leg4_traj[:, 19:22], body_traj[:, 22:25]))

    #     gait_angles_file_path = Path(
    #         f'gait_angles/gait_angles_DIR_{DIR}_WP{WAYPOINTS}_START_HALF_STEP_{strftime("%Y%m%d_%H%M%S")}.npy')
    #     gait_angles_file_path.parent.mkdir(parents=True, exist_ok=True)
    #     np.save(gait_angles_file_path, q)

    #     # if hexy.viz_flag:
    #     #     hexy.viz.play(q)

    #     # Update the current configuration
    #     hexy.qc = q[-1]
    #     step_size_xy_mult = 2
    #     # Generate trajectories for legs 1, 3, and 5
    #     leg1_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=1, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     leg3_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=3, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     leg5_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=5, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     # Generate body trajectory
    #     body_traj = hexy.generate_body_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     # Append trajectories to q
    #     q = np.vstack((q,
    #                    np.hstack((body_traj[:, 0:7], body_traj[:, 7:10],
    #                               leg1_traj[:, 10:13], body_traj[:, 13:16],
    #                               leg3_traj[:, 16:19], body_traj[:, 19:22], leg5_traj[:, 22:25]))
    #                    ))
    #     # Update the current configuration
    #     hexy.qc = q[-1]
    #     # Generate trajectories for legs 0, 2, and 4 again
    #     leg0_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=0, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     leg2_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=2, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     leg4_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=4, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     # Generate body trajectory
    #     body_traj = hexy.generate_body_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     # Append trajectories to q
    #     q = np.vstack((q,
    #                    np.hstack((body_traj[:, 0:7], leg0_traj[:, 7:10],
    #                               body_traj[:, 10:13], leg2_traj[:, 13:16],
    #                               body_traj[:, 16:19], leg4_traj[:, 19:22], body_traj[:, 22:25]))
    #                    ))
    #     # Update the current configuration
    #     hexy.qc = q[-1]

    #     gait_angles_file_path = Path(
    #         f'gait_angles/gait_angles_DIR_{DIR}_WP{WAYPOINTS}_MID_FULL_STEP_{strftime("%Y%m%d_%H%M%S")}.npy')
    #     gait_angles_file_path.parent.mkdir(parents=True, exist_ok=True)
    #     np.save(gait_angles_file_path, q)

    #     # if hexy.viz_flag:
    #     #     hexy.viz.play(q)

    #     step_size_xy_mult = 1
    #     # Generate trajectories for legs 1, 3, and 5
    #     leg1_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=1, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     leg3_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=3, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     leg5_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, LEG=5, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     # Generate body trajectory
    #     body_traj = hexy.generate_body_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, DIR=DIR, t_goal=t_goal, WAYPOINTS=WAYPOINTS)
    #     # Append trajectories to q
    #     q = np.vstack((q,
    #                    np.hstack((body_traj[:, 0:7], body_traj[:, 7:10],
    #                               leg1_traj[:, 10:13], body_traj[:, 13:16],
    #                               leg3_traj[:, 16:19], body_traj[:, 19:22], leg5_traj[:, 22:25]))
    #                    ))
    #     # Update the current configuration
    #     q = np.vstack((q, np.hstack((q[-1, :7], np.zeros(hexy.robot.nq - 7)))))

    #     gait_angles_file_path = Path(
    #         f'gait_angles/gait_angles_DIR_{DIR}_WP{WAYPOINTS}_END_HALF_STEP_{strftime("%Y%m%d_%H%M%S")}.npy')
    #     gait_angles_file_path.parent.mkdir(parents=True, exist_ok=True)
    #     np.save(gait_angles_file_path, q)

    #     # hexy.plot_trajctory(state=np.array([hexy.forward_kinematics(
    #     #     q=qi) for qi in q]), title=f'Start Half Step + Two Full Steps + Stop Half Step in Direction = {DIR}')
    #     if hexy.viz_flag:
    #         hexy.viz.play(q)
    #     sleep(3)
    #     # hexy.qc = hexy.robot.q0
    #     hexy.qc = q[-1]
    #     sleep(3)
