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
from warnings import warn

# We don't want to print every decimal!
np.set_printoptions(suppress=True, precision=4)


class hexapod:
    """
    A class representing a hexapod robot model using the Pinocchio library.

    This class provides methods to initialize the robot model, perform inverse kinematics,
    generate trajectories, and visualize the robot's movements.
    """

    def __init__(self, init_viz: bool = False, logging_level: int = logging.WARN) -> None:
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
        self.HALF_STEP_SIZE_XY = 0.05/2  # Half of the step size in XY plane
        self.XY_CLEARANCE = 0.015
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
        p2 = p1[0:2] + (self.HALF_STEP_SIZE_XY * step_size_xy_mult * v * 2)
        # Define trajectory functions for x(t) and y(t) as linear interpolation between p1 and p2
        p2 = np.append(p2, p1[2])

        # Apex points (midway)
        y_apex = max(p1[1], p2[1]) + (self.XY_CLEARANCE *
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

    def generate_waypoints(self, step_size_xy_mult: float, WAYPOINTS: int = 5, DIR: str = 'N', leg_set: int = 0):
        start = None
        if (leg_set == 0 or leg_set == 1):
            start = leg_set
        else:
            raise ValueError(
                f'Expected value for leg_set is a string "024" or "135". Got {leg_set} instead')

        waypoints = self.state_c @ np.ones((1, WAYPOINTS))

        for foot in self.FOOT_IDS[start::2]:
            self.init_foot_trajectory_functions(
                step_size_xy_mult=step_size_xy_mult, DIR=DIR, FOOT_ID=foot)
            # Create time steps from 0 to 1
            s = np.linspace(0, 1, WAYPOINTS)
            # Generate waypoints by evaluating trajectory functions at each time step
            wp = np.array([[round(self.x_t(t), 5), round(
                self.y_t(t), 5), round(self.z_t(t), 5)] for t in s]).T
            idx = (3 * (foot // 8 - 1) + 3)
            waypoints[idx:idx+3, :] = wp

        # Initialize body trajectory functions
        self.init_body_trajectory_functions(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR)
        s = np.linspace(0, 1, WAYPOINTS)
        # Generate waypoints by evaluating trajectory functions at each time step
        wp = np.array([[round(self.x_t(t), 5), round(
            self.y_t(t), 5), self.state_c[2][0]] for t in s]).T
        waypoints[0:3, :] = wp
        return waypoints

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

    def _map_to_full(self, J_trans: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        J_trans: (3×24) translational Jacobian in minimal coords:
          cols 0–2 = body xyz, 3–5 = minimal orientation, 6–23 = 18 joints
        Returns J_full: (3×25) in your full q = [x,y,z,qx,qy,qz,qw,θ1…θ18].
        """
        # split out the blocks
        J_xyz = J_trans[:, :3]             # (3×3)
        J_dtheta = J_trans[:, 3:6]       # (3×3) minimal orient
        J_theta = J_trans[:, 6:]        # (3×18)

        # build the 4×3 Omega matrix for quaternion q = [x,y,z, qx,qy,qz,qw, …]
        qx, qy, qz, qw = q[3], q[4], q[5], q[6]
        Ω = np.array([
            [-qx, -qy, -qz],
            [qw, -qz,  qy],
            [qz,  qw, -qx],
            [-qy,  qx,  qw],
        ])  # (4×3)

        # M = ∂δθ/∂q_quat = 2 * Ω^T  (3×4)
        M = 2 * Ω.T

        # map the 3 minimal‐orient cols into 4 quaternion cols
        J_quat_full = J_dtheta.dot(M)    # (3×4)

        # stack: [ 3×3 | 3×4 | 3×18 ] → (3×25)
        return np.hstack([J_xyz, J_quat_full, J_theta])

    def compute_full_fk_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Build the 21×25 Jacobian of FK(q) by stacking each frame's
        3×25 full Jacobian.
        """
        J_blocks = []
        for fid in [self.BASE_FRAME_ID] + self.FOOT_IDS:
            J6 = pin.computeFrameJacobian(
                self.robot.model, self.robot.data, q, fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )  # (6×24)
            J_trans = J6[:3, :]             # (3×24)
            J_full = self._map_to_full(J_trans, q)  # (3×25)
            J_blocks.append(J_full)
        return np.vstack(J_blocks)          # (21×25)

    def cost(self, q: np.ndarray, desired_pos: np.ndarray) -> float:
        Q = np.eye(21)
        x = self.forward_kinematics(q)    # (21×1)
        diff = x - desired_pos               # (21×1)
        R = np.eye(self.robot.nq)
        # return 0.5*(diff.T @ Q @ diff + q.T @ R @ q).item()
        return 0.5*(diff.T @ Q @ diff).item()

    def cost_grad(self, q: np.ndarray, desired_pos: np.ndarray) -> np.ndarray:
        Q = np.eye(21)
        x = self.forward_kinematics(q)         # (21×1)
        diff = x - desired_pos                    # (21×1)
        J_fk = self.compute_full_fk_jacobian(q)   # (21×25)
        if (np.isnan(J_fk).any()):
            print('NaN in Jacobian')
        grad_fk = J_fk.T @ (Q @ diff)              # (25×1)
        return grad_fk.flatten()        # (25,)

    def eq_constraints(self, q: np.ndarray) -> np.ndarray:
        """
        Enforce z_body=0, qx=0, qy=0, qz=0, qw=1.
        Returns c(q) in R^5, to be driven to zero.
        """
        desired = np.array([0., 0., 0., 0., 1.])
        actual = q[2:7]   # [z_body, qx, qy, qz, qw]
        return actual - desired

    def eq_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Jacobian of eq_constraints: ∂c_i/∂q_j, shape (5×25).
        """
        J = np.zeros((5, self.robot.nq))
        J[0, 2] = 1.0   # ∂(q2)/∂q2
        J[1, 3] = 1.0
        J[2, 4] = 1.0
        J[3, 5] = 1.0
        J[4, 6] = 1.0
        return J

    def optimize(self, q0: np.ndarray, desired_pos: np.ndarray):
        cons = {
            'type': 'eq',
            'fun':  self.eq_constraints,
            'jac':  self.eq_jacobian
        }
        res = minimize(
            fun=self.cost,
            x0=q0,
            args=(desired_pos,),
            jac=self.cost_grad,
            constraints=cons,
            method='SLSQP',     # or 'trust-constr'
            options={'ftol': 1e-9, 'maxiter': 500, 'disp': True},
            tol=1e-8
        )
        return res

    def mpc_cost(self, q: np.ndarray, desired_seq: List[np.ndarray], horizon: int):
        n_q = self.robot.nq
        q_seq_flat = q.flatten()
        total_cost = 0.0
        # Loop over each step in the horizon
        for i in range(horizon):
            # Extract the i-th configuration from the flattened decision vector
            q_i = q_seq_flat[i*n_q:(i+1)*n_q]
            total_cost += self.cost(q=q_i, desired_pos=desired_seq[i])
        return total_cost

    def mpc_cost_grad(self, q: np.ndarray, desired_seq: List[np.ndarray], horizon: int):
        n_q = self.robot.nq
        q_seq_flat = q.flatten()
        grad_seq = np.zeros(q_seq_flat.shape)
        for i in range(horizon):
            # Extract the i-th configuration from the flattened decision vector.
            q_i = q_seq_flat[i * n_q:(i + 1) * n_q]
            grad_seq[i * n_q:(i + 1) * n_q] = self.cost_grad(q=q_i,
                                                             desired_pos=desired_seq[i])
        if (np.isnan(grad_seq).any()):
            print('NaN in gradient')
        return grad_seq

    def mpc_eq_constraints(self, q_seq_flat: np.ndarray, horizon: int) -> np.ndarray:
        """
        Construct equality constraints for each configuration in the MPC horizon.
        For each predicted configuration q_i (of dimension n_q), enforce that:
            q_i[2:7] == [0, 0, 0, 0, 1]
        The constraints are stacked into one long vector.

        Args:
            q_seq_flat (np.ndarray): Flattened decision vector of shape (horizon*n_q,).
            horizon (int): The prediction horizon (number of configurations).

        Returns:
            np.ndarray: Concatenated constraint vector (length 5*horizon) that should equal zero.
        """
        n_q = self.robot.nq
        constraints = []
        for i in range(horizon):
            constraints.append(self.eq_constraints(
                q=q_seq_flat[i * n_q:(i + 1) * n_q]))
        return np.concatenate(constraints)

    def mpc_eq_jacobian(self, q_seq_flat: np.ndarray, horizon: int) -> np.ndarray:
        """
        Compute the Jacobian of the equality constraints over the entire MPC horizon.
        Each block is the Jacobian of a single configuration's constraints (5×n_q),
        arranged in a block-diagonal structure of size (5*horizon, n_q*horizon).

        Args:
            q_seq_flat (np.ndarray): Flattened decision vector of shape (horizon*n_q,).
            horizon (int): Prediction horizon.

        Returns:
            np.ndarray: Block-diagonal Jacobian matrix of shape (5*horizon, n_q*horizon).
        """
        n_q = self.robot.nq
        # Get the Jacobian for a single configuration.
        eq_jac_single = self.eq_jacobian(q_seq_flat[:n_q])  # shape: (5, n_q)
        # Build block diagonal matrix by repeating eq_jac_single along the diagonal.
        return np.kron(np.eye(horizon), eq_jac_single)

    def mpc_step(self, current_q: np.ndarray, desired_seq: List[np.ndarray], horizon: int):
        n_q = self.robot.nq
        # Initial guess: repeat current_q over the horizon
        q_guess = np.tile(current_q, horizon)

        # Define the equality constraints for the optimizer.
        cons = [{
            'type': 'eq',
            'fun': lambda q_seq: self.mpc_eq_constraints(q_seq, horizon),
            'jac': lambda q_seq: self.mpc_eq_jacobian(q_seq, horizon)
        }]

        # Define the optimization problem using your new MPC cost function
        res = minimize(
            fun=lambda q_seq: self.mpc_cost(q_seq, desired_seq, horizon),
            x0=q_guess,
            jac=lambda q_seq: self.mpc_cost_grad(q_seq, desired_seq, horizon),
            constraints=cons,
            method='SLSQP',   # or another suitable method
            options={'ftol': 1e-9, 'maxiter': 500, 'disp': True}
        )
        # Reshape the solution if needed
        q_seq_opt = res.x.reshape(horizon, n_q)
        # Return only the first configuration to apply to the robot
        return q_seq_opt[0]

    def update_current_pose(self, q: np.ndarray):
        if q.shape != (25, ):
            raise ValueError(
                f'Expected shape of parameter q was (25, ), but got {q.shape}')
        self.qc = q
        self.state_c = self.forward_kinematics(q=q)


if __name__ == "__main__":

    # Create a hexapod instance with visualization and debug logging
    hexy = hexapod(init_viz=False, logging_level=logging.WARNING)

    # q_old = np.load('gait_angles/gait_angles_DIR_N_WP5_S1_20250406_134944.npy')
    # states = np.array([hexy.forward_kinematics(q_i) for q_i in q_old])
    # hexy.plot_trajctory(state=states, title='q old')
    # exit()
    # sleep(3)
    # Set parameters for movement
    v = 0.5  # Velocity in m/s
    # start_time = time()
    WAYPOINTS = 20
    # DIR = 'N'
    # start = time()
    q = np.copy(hexy.qc)

    horizon = 3

    wp = hexy.generate_waypoints(
        WAYPOINTS=WAYPOINTS, step_size_xy_mult=1, leg_set=0)
    wp = [wp[:, i].reshape(-1, 1) for i in range(wp.shape[1])]
    for i in range(len(wp)):
        window = wp[i:i + horizon]
        if len(window) < horizon:
            # pad with last element
            window += [wp[-1]] * (horizon - len(window))
        start = time()
        qi = hexy.mpc_step(current_q=hexy.qc,
                           desired_seq=window, horizon=horizon)
        print(f'Optimized in {time()-start}s')
        hexy.update_current_pose(q=qi)
        q = np.vstack((q, qi))

    wp = hexy.generate_waypoints(
        WAYPOINTS=WAYPOINTS, step_size_xy_mult=2, leg_set=1)
    wp = [wp[:, i].reshape(-1, 1) for i in range(wp.shape[1])]
    for i in range(len(wp)):
        window = wp[i:i + horizon]
        if len(window) < horizon:
            # pad with last element
            window += [wp[-1]] * (horizon - len(window))
        start = time()
        qi = hexy.mpc_step(current_q=hexy.qc,
                           desired_seq=window, horizon=horizon)
        print(f'Optimized in {time()-start}s')
        hexy.update_current_pose(q=qi)
        q = np.vstack((q, qi))

    wp = hexy.generate_waypoints(
        WAYPOINTS=WAYPOINTS, step_size_xy_mult=2, leg_set=0)
    wp = [wp[:, i].reshape(-1, 1) for i in range(wp.shape[1])]
    for i in range(len(wp)):
        window = wp[i:i + horizon]
        if len(window) < horizon:
            # pad with last element
            window += [wp[-1]] * (horizon - len(window))
        start = time()
        qi = hexy.mpc_step(current_q=hexy.qc,
                           desired_seq=window, horizon=horizon)
        print(f'Optimized in {time()-start}s')
        hexy.update_current_pose(q=qi)
        q = np.vstack((q, qi))

    wp = hexy.generate_waypoints(
        WAYPOINTS=WAYPOINTS, step_size_xy_mult=1, leg_set=1)
    wp = [wp[:, i].reshape(-1, 1) for i in range(wp.shape[1])]
    for i in range(len(wp)):
        window = wp[i:i + horizon]
        if len(window) < horizon:
            # pad with last element
            window += [wp[-1]] * (horizon - len(window))
        start = time()
        qi = hexy.mpc_step(current_q=hexy.qc,
                           desired_seq=window, horizon=horizon)
        print(f'Optimized in {time()-start}s')
        hexy.update_current_pose(q=qi)
        q = np.vstack((q, qi))
    q = np.delete(q, 0, axis=0)
    states = np.array([hexy.forward_kinematics(q_i) for q_i in q])
    hexy.plot_trajctory(state=states, title='v2.5.3 MPC')

    # wp_hist = wp
    # # hexy.plot_trajctory(state=wp.T, title='traj')
    # q = np.copy(hexy.qc)
    # # q = np.zeros(0)
    # for i in range(WAYPOINTS):
    #     qi = hexy.optimize(
    #         q0=hexy.qc, desired_pos=wp[:, i].reshape(-1, 1)).x
    #     hexy.qc = qi
    #     hexy.state_c = hexy.forward_kinematics(hexy.qc)
    #     q = np.vstack((q, qi))
    # # print(f'Computation done in {time()-start} seconds')
    # q = np.delete(q, 0, axis=0)
    # # states = np.array([hexy.forward_kinematics(q_i) for q_i in q])
    # # hexy.plot_trajctory(state=states, title='v2.5.1')

    # # start = time()
    # wp = hexy.generate_waypoints(
    #     WAYPOINTS=WAYPOINTS, step_size_xy_mult=2, leg_set=1)
    # wp_hist = np.hstack((wp_hist, wp))
    # # hexy.plot_trajctory(state=wp.T, title='traj')
    # for i in range(WAYPOINTS):
    #     qi = hexy.optimize(
    #         q0=hexy.qc, desired_pos=wp[:, i].reshape(-1, 1)).x
    #     hexy.qc = qi
    #     hexy.state_c = hexy.forward_kinematics(hexy.qc)
    #     q = np.vstack((q, qi))
    # # print(f'Computation done in {time()-start} seconds')
    # # states = np.array([hexy.forward_kinematics(q_i) for q_i in q])
    # # hexy.plot_trajctory(state=states, title='v2.5.1')

    # # start = time()
    # wp = hexy.generate_waypoints(
    #     WAYPOINTS=WAYPOINTS, step_size_xy_mult=2, leg_set=0)
    # wp_hist = np.hstack((wp_hist, wp))
    # for i in range(WAYPOINTS):
    #     qi = hexy.optimize(
    #         q0=hexy.qc, desired_pos=wp[:, i].reshape(-1, 1)).x
    #     hexy.qc = qi
    #     hexy.state_c = hexy.forward_kinematics(hexy.qc)
    #     q = np.vstack((q, qi))
    # # print(f'Computation done in {time()-start} seconds')
    # # states = np.array([hexy.forward_kinematics(q_i) for q_i in q])
    # # hexy.plot_trajctory(state=states, title='v2.5.1')

    # # start = time()
    # wp = hexy.generate_waypoints(
    #     WAYPOINTS=WAYPOINTS, step_size_xy_mult=1, leg_set=1)
    # wp_hist = np.hstack((wp_hist, wp))
    # for i in range(WAYPOINTS):
    #     qi = hexy.optimize(
    #         q0=hexy.qc, desired_pos=wp[:, i].reshape(-1, 1)).x
    #     hexy.qc = qi
    #     hexy.state_c = hexy.forward_kinematics(hexy.qc)
    #     q = np.vstack((q, qi))
    # print(f'Computation done in {time()-start} seconds')
    # q = np.delete(q, 0, axis=0)
    # hexy.viz.play(q)
    # states = np.array([hexy.forward_kinematics(q_i) for q_i in q])
    # # hexy.plot_trajctory(state=states, title='v2.5.3')
    # # hexy.plot_trajctory(state=wp_hist.T, title='Waypoints')
    # # comp = q == q_old
