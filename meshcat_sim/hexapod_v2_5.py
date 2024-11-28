# We import useful libraries
from time import sleep, time, strftime
import numpy as np
import pinocchio as pin
import meshcat.geometry as g
from pathlib import Path
import sys
from scipy.optimize import minimize
import logging
import matplotlib.pyplot as plt

# We don't want to print every decimal!
np.set_printoptions(suppress=True, precision=4)


class hexapod:
    """
    A class representing a hexapod robot model using the Pinocchio library.

    This class provides methods to initialize the robot model, perform inverse kinematics,
    generate trajectories, and visualize the robot's movements.
    """

    def __init__(self, init_viz=True, logging_level=logging.DEBUG) -> None:
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
        # Set up movement parameters
        self.direction_slope = 0.0
        self.HALF_STEP_SIZE_XY = 0.06 / 2  # Half of the step size in XY plane
        self.STEP_SIZE_Z = 0.01  # Step size in Z (vertical) direction
        # Retrieve frame IDs for the feet
        self.FOOT_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "foot" in frame.name]
        # Retrieve frame IDs for the shoulder joints
        self.SHOULDER_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "Revolute_joint_0" in frame.name]
        # Get the frame ID for the robot base
        self.BASE_FRAME_ID = self.robot.model.getFrameId("robot_base")
        # Set initial state and configuration
        self.state_flag = 'START'
        self.qc = self.robot.q0  # Initial joint configuration
        # Initialize visualization if requested
        self.viz_flag = init_viz
        self.theta = np.pi / 4
        if self.viz_flag == True:
            self.init_viz()
        self.L_of = 0.058
        self.phi = np.pi/4
        for frame in self.robot.model.frames:
            # Print the frame name, type, ID, and position
            self.logger.info(
                f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")

        # Log successful initialization
        self.logger.info(
            f"Hexapod Object Initialised Successfully with init_viz = {self.viz_flag}, logging_level={logging_level}")

    def init_logger(self, logging_level):
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
        log_file = f'hexapod_log_{timestamp}.log'
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

    def init_viz(self):
        """
        Initialize the Meshcat visualizer for the robot.
        """
        try:
            # Initialize Meshcat visualizer
            self.viz = pin.visualize.MeshcatVisualizer(
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

    def print_all_frame_info(self):
        """
        Print information about all frames in the robot model.
        """
        for frame in self.robot.model.frames:
            # Print the frame name, type, ID, and position
            print(
                f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")

    def find_frame_info_by_name(self, name: str):
        """
        Find and print information about frames matching the given name.

        Args:
            name (str): Substring to match in frame names.
        """
        for frame in self.robot.model.frames:
            if name in frame.name:
                # Print the frame name, type, ID, and position
                print(
                    f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")

    def forward_kinematics(self, q):
        return np.concatenate((self.robot.framePlacement(q=q, index=self.BASE_FRAME_ID).translation, np.array([self.robot.framePlacement(q=q, index=foot_id).translation for foot_id in self.FOOT_IDS]).flatten()))

    def foot_pos_err(self, q, FRAME_ID=9, desired_pos=np.zeros(3)):
        self.robot.forwardKinematics(q)
        current_pos = self.robot.framePlacement(q, FRAME_ID).translation
        error = current_pos - desired_pos
        # Optionally apply weights
        weights = np.array([2.0, 3.0, 5.0])  # Modify as needed
        weighted_error = weights * error
        return np.dot(weighted_error, weighted_error)

    def inverse_geometery(self, q, FRAME_ID=9, desired_pos=np.zeros(3)):
        """
        Perform inverse kinematics to find joint configuration that places a frame at the desired position.

        Args:
            q (numpy.ndarray): Initial guess for the joint configuration.
            FRAME_ID (int): Frame ID for which to compute inverse kinematics.
            desired_pos (numpy.ndarray): Desired position for the frame.

        Returns:
            numpy.ndarray: Joint configuration that minimizes the positional error.
        """
        # # Set bounds for optimization (fix the base joint positions)
        # bounds = [(0., 0.)]*self.robot.nq
        # bounds[0] = (self.qc[0], self.qc[0])  # x position of base
        # bounds[1] = (self.qc[1], self.qc[1])  # y position of base
        # bounds[2] = (self.qc[2], self.qc[2])  # z position of base
        # bounds[3] = (self.qc[3], self.qc[3])  # Quaternion component
        # bounds[4] = (self.qc[4], self.qc[4])  # Quaternion component
        # bounds[5] = (self.qc[5], self.qc[5])  # Quaternion component
        # bounds[6] = (self.qc[6], self.qc[6])  # Quaternion component
        # for i in range(7, self.robot.nq):
        #     # Joint limits for other joints
        #     bounds[i] = ((-np.pi/3), (np.pi/3))
        # # Perform minimization to find joint configuration minimizing foot position error

        bounds = [((-np.pi / 3, np.pi / 3) if int(7 + (3 * (np.floor(FRAME_ID / 8) - 1))) <= i < int(7 +
                   (3 * (np.floor(FRAME_ID / 8) - 1))) + 3 else (self.qc[i], self.qc[i])) for i in range(self.robot.nq)]

        res = minimize(
            self.foot_pos_err, q, args=(FRAME_ID, desired_pos), bounds=bounds, tol=1e-12, method='L-BFGS-B')
        # Return the optimized joint configuration
        return res.x

    def north_vector(self):
        """
        Compute the unit vector pointing in the 'north' direction based on the robot's configuration.

        Returns:
            numpy.ndarray: Unit vector in the 'north' direction.
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

    def south_vector(self):
        """
        Compute the unit vector pointing in the 'south' direction based on the robot's configuration.

        Returns:
            numpy.ndarray: Unit vector in the 'south' direction.
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

    def east_vector(self):
        """
        Placeholder method for computing the 'east' direction vector.
        """
        pass

    def west_vector(self):
        """
        Placeholder method for computing the 'west' direction vector.
        """
        pass

    def north_east_vector(self):
        """
        Placeholder method for computing the 'northeast' direction vector.
        """
        pass

    def north_west_vector(self):
        """
        Placeholder method for computing the 'northwest' direction vector.
        """
        pass

    def south_east_vector(self):
        """
        Placeholder method for computing the 'southeast' direction vector.
        """
        pass

    def south_west_vector(self):
        """
        Placeholder method for computing the 'southwest' direction vector.
        """
        pass

    def default_vector(self):
        """
        Raise an error for invalid direction inputs.
        """
        raise KeyError(
            "NOT A VAILD DIRECTION. CHOOSE ONE OF {N, S, E, W, NE, NW, SE, SW} ONLY")

    def generate_direction_vector(self, DIR='N'):
        """
        Generate a unit vector corresponding to the specified direction.

        Args:
            DIR (str): Direction string, one of {'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'}.

        Returns:
            numpy.ndarray: Unit vector in the specified direction.
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

    def init_foot_trajectory_functions(self, step_size_xy_mult, DIR='N', FOOT_ID=10):
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

        y_clearance = 0.03           # Vertical clearance for y apex
        z_clearance = 0.03           # Lateral deviation for z apex

        # Apex points (midway)
        x_apex = (p1[0] + p2[0]) / 2
        y_apex = max(p1[1], p2[1]) + (y_clearance * np.sign(max(p1[1], p2[1])))
        z_apex = max(p1[2], p2[2]) + (z_clearance * np.sign(max(p1[2], p2[2])))

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

        # Parameterize t
        # t = np.linspace(0, 1, 100)

        # Generate 3D trajectory
        # Linear interpolation for x
        self.x_t = lambda t: (1 - t) * p1[0] + t * p2[0]
        self.y_t = lambda t: a_y * t**2 + b_y * t + c_y
        self.z_t = lambda t: a_z * t**2 + b_z * t + c_z

    def init_body_trajectory_functions(self, step_size_xy_mult, DIR='N'):

        # Generate direction vector based on specified direction
        v = self.generate_direction_vector(DIR=DIR)
        # Get initial position of the foot in XY plane
        p1 = self.robot.framePlacement(self.qc, self.BASE_FRAME_ID).translation
        # Compute the target position p2 in XY plane
        p2 = p1[0:2] + (self.HALF_STEP_SIZE_XY * step_size_xy_mult * v)
        # Define trajectory functions for x(t) and y(t) as linear interpolation between p1 and p2
        p2 = np.append(p2, p1[2])
        self.x_t = lambda t: ((1 - t) * p1[0]) + (t * p2[0])
        self.y_t = lambda t:  ((1 - t) * p1[1]) + (t * p2[1])

        # Define trajectory function for z(t) as a parabolic curve (quadratic)
        self.z_t = lambda t: (((-self.STEP_SIZE_Z)/0.25) *
                              (t ** 2)) + ((self.STEP_SIZE_Z / 0.25) * t)

    def generate_joint_waypoints(self, step_size_xy_mult, STEPS=5, DIR='N', FOOT_ID=10):
        """
        Generate joint waypoints for a foot trajectory.

        Args:
            step_size_xy_mult (float): Multiplier for step size in XY plane.
            STEPS (int): Number of steps in the trajectory.
            DIR (str): Direction of movement ('N', 'S', etc.).
            FOOT_ID (int): Frame ID of the foot.

        Returns:
            list: List of joint configurations along the trajectory.
        """
        # Initialize foot trajectory functions
        self.init_foot_trajectory_functions(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, FOOT_ID=FOOT_ID)
        # Create time steps from 0 to 1
        s = np.linspace(0, 1, STEPS)
        # Generate waypoints by evaluating trajectory functions at each time step
        waypoints = [[round(self.x_t(t), 5), round(
            self.y_t(t), 5), round(self.z_t(t), 5)] for t in s]
        # Convert waypoints to array for visualization
        points = np.array([waypoints[0], waypoints[-1]]).T

        # self.plot_trajctory(title=f'Trajectory for Leg {(np.floor(FOOT_ID / 8) - 1)} Before IK ', state=np.array([np.concatenate([np.zeros(int(3 + (3 * (np.floor(FOOT_ID / 8) - 1)))), np.array(waypoints)[
        #                     i, :], np.zeros(self.robot.nq - int(3 + (3 * (np.floor(FOOT_ID / 8) - 1)) + 3))]) for i in range(len(waypoints))]))

        # Visualize the foot trajectory if visualization is enabled
        if self.viz_flag:
            self.viz.viewer[('Foot_ID_' + str(FOOT_ID) + '_trajectory')].set_object(
                g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xffff00)))

        # Perform inverse kinematics to get joint configurations for each waypoint
        return [self.inverse_geometery(self.qc, FOOT_ID, wp)
                for wp in waypoints]

    def compute_trajectory_pva(self, position_init, position_goal, t_init, t_goal, t):
        """Time parameterization of the trajectory with acceleration and velocity constraints

        Args:
            position_init (numpy.ndarray): Initial position in Joint state form
            position_goal (numpy.ndarray): Goal position in Joint state form
            t_init (float): Initial time
            t_goal (float): Goal time
            t (float): Current time

        Returns:
            tuple: Desired position, velocity, and acceleration at time t
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

    def compute_trajectory_pv(self, position_init, position_goal, t_init, t_goal, t):
        """
        Compute the desired trajectory, velocity, and acceleration at time t
        using third-degree (cubic) polynomial equations.

        Args:
            position_init (numpy.ndarray): Initial position in Joint state form
            position_goal (numpy.ndarray): Goal position in Joint state form
            t_init (float): Initial time
            t_goal (float): Goal time
            t (float): Current time

        Returns:
            tuple: Desired position, velocity, and acceleration at time t
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

    def compute_trajectory_p(self, position_init, position_goal, t_init, t_goal, t):
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

    def generate_leg_joint_trajectory(self, step_size_xy_mult, DIR='N', LEG=0, STEPS=5, t_init=0, t_goal=0.1, dt=0.01):
        """
        Generate a joint trajectory for a specific leg.

        Args:
            step_size_xy_mult (float): Multiplier for step size in XY plane.
            DIR (str): Direction of movement ('N', 'S', etc.).
            LEG (int): Index of the leg (0 to 5).
            STEPS (int): Number of steps in the trajectory.
            t_init (float): Initial time.
            t_goal (float): Goal time.
            dt (float): Time step for trajectory generation.

        Returns:
            numpy.ndarray: Array of joint configurations along the trajectory.
        """
        # Generate joint waypoints for the specified leg
        q_wps = self.generate_joint_waypoints(step_size_xy_mult,
                                              DIR=DIR, FOOT_ID=self.FOOT_IDS[LEG], STEPS=STEPS)
        # self.plot_trajctory(title=f'Trajectory for Leg {LEG} after IK Before Interpolation', state=np.array(
        #     [self.forward_kinematics(q) for q in q_wps]))
        q_traj = self.qc
        # Create a mask to apply joint configurations to the specific leg
        mask = np.concatenate((np.zeros(6), [1], np.zeros(
            LEG*3), [1, 1, 1], np.zeros((5-LEG)*3)))
        for i in range(0, q_wps.__len__()-1):
            t = t_init
            while t <= t_goal:
                # Compute trajectory using linear interpolation
                q_t = self.compute_trajectory_p(
                    q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
                q_traj = np.vstack((q_traj, np.multiply(q_t, mask)))
                t = (t + dt)
        # Remove the initial configuration

        # self.plot_trajctory(
        #     np.array([self.forward_kinematics(q) for q in q_traj]), title=f'Leg joint traj for Leg {LEG} After Interpolation')
        return np.delete(q_traj, 0, axis=0)

    def get_foot_positions(self, q):
        """
        Get the positions of all feet for a given joint configuration.

        Args:
            q (numpy.ndarray): Joint configuration vector.

        Returns:
            list: List of foot positions.
        """
        return [self.robot.framePlacement(q, foot_id).translation for foot_id in self.FOOT_IDS]

    def feet_error(self, q_joints, desired_base_pose):
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
        error = 0
        for foot_id, desired_pos in zip(self.FOOT_IDS, initial_foot_positions):
            current_pos = self.robot.framePlacement(
                q_full, foot_id).translation
            error += np.linalg.norm(current_pos - desired_pos)**2
        return error

    def body_inverse_geometry(self, q, desired_base_pos):
        """
        Compute inverse kinematics for the robot's body to maintain foot positions.

        Args:
            q (numpy.ndarray): Current joint configuration.
            desired_base_pos (numpy.ndarray): Desired base position.

        Returns:
            numpy.ndarray: Joint configuration that maintains foot positions.
        """
        # Joint angle bounds (exclude base joints)
        bounds = [(-np.pi/3, np.pi/3)] * (self.robot.nq - 7)

        # Initial joint angles
        q_joints_init = q[7:].copy()

        res = minimize(
            self.feet_error,
            q_joints_init, args=(desired_base_pos),
            bounds=bounds,
            method='L-BFGS-B', options={'disp': False},
            tol=1e-10
        )

        return np.concatenate([desired_base_pos, res.x])

    def generate_body_path_waypoints(self, step_size_xy_mult=1, STEPS=5, DIR='N'):
        """
        Generate waypoints for the robot body's path.

        Args:
            step_size_xy_mult (float): Multiplier for step size in XY plane.
            STEPS (int): Number of steps in the trajectory.
            DIR (str): Direction of movement ('N', 'S', etc.).

        Returns:
            list: List of joint configurations along the body's path.
        """
        # Initialize foot trajectory functions for the base
        self.init_body_trajectory_functions(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR)
        s = np.linspace(0, 1, STEPS)
        # Generate waypoints by evaluating trajectory functions at each time step
        waypoints = [np.concatenate(([round(self.x_t(t), 5), round(
            self.y_t(t), 5)], self.qc[2:7].copy())) for t in s]

        points = np.array(waypoints)[:, 0:3].T
        # Visualize the base trajectory if visualization is enabled
        if self.viz_flag:
            self.viz.viewer[('Base_trajectory')].set_object(
                g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xffff00)))
        # Perform inverse kinematics to get joint configurations for each waypoint
        return [self.body_inverse_geometry(self.qc, wp)
                for wp in waypoints]

    def generate_body_joint_trajectory(self, step_size_xy_mult, DIR='N', STEPS=5, t_init=0, t_goal=0.1, dt=0.01):
        """
        Generate a joint trajectory for the robot's body.

        Args:
            step_size_xy_mult (float): Multiplier for step size in XY plane.
            DIR (str): Direction of movement ('N', 'S', etc.).
            STEPS (int): Number of steps in the trajectory.
            t_init (float): Initial time.
            t_goal (float): Goal time.
            dt (float): Time step for trajectory generation.

        Returns:
            numpy.ndarray: Array of joint configurations along the trajectory.
        """
        # Generate waypoints for the body's path
        q_wps = self.generate_body_path_waypoints(step_size_xy_mult,
                                                  DIR=DIR, STEPS=STEPS)
        # self.plot_trajctory(
        #     np.array([self.forward_kinematics(q) for q in q_wps]))
        q_traj = self.qc
        for i in range(0, q_wps.__len__()-1):
            t = t_init
            while t <= t_goal:
                # Compute trajectory using linear interpolation
                q_t = self.compute_trajectory_p(
                    q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
                q_traj = np.vstack((q_traj, q_t))
                t = (t + dt)
        # Remove the initial configuration
        return np.delete(q_traj, 0, axis=0)

    def compute_gait(self, v: float = 0.5, STEPS: int = 5, STEP_CNT: int = 3):
        step_size_xy_mult = 1
        t_goal = self.HALF_STEP_SIZE_XY / v
        start_time = time()
        # Generate trajectories for legs 0, 2, and 4
        leg0_traj = self.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=0, t_goal=t_goal, STEPS=STEPS)
        leg2_traj = self.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=2, t_goal=t_goal, STEPS=STEPS)
        leg4_traj = self.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=4, t_goal=t_goal, STEPS=STEPS)
        # Generate body trajectory
        body_traj = self.generate_body_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', t_goal=t_goal, STEPS=STEPS)
        # Combine the trajectories into a single trajectory q
        q = np.hstack((body_traj[:, 0:7], leg0_traj[:, 7:10], body_traj[:, 10:13],
                       leg2_traj[:, 13:16], body_traj[:, 16:19], leg4_traj[:, 19:22], body_traj[:, 22:25]))
        # Update the current configuration
        self.qc = q[-1]
        step_size_xy_mult = 2
        for i in range(0, STEP_CNT):
            # Generate trajectories for legs 1, 3, and 5
            leg1_traj = self.generate_leg_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=1, t_goal=t_goal, STEPS=STEPS)
            leg3_traj = self.generate_leg_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=3, t_goal=t_goal, STEPS=STEPS)
            leg5_traj = self.generate_leg_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=5, t_goal=t_goal, STEPS=STEPS)
            # Generate body trajectory
            body_traj = self.generate_body_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR='N', t_goal=t_goal, STEPS=STEPS)
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
                step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=0, t_goal=t_goal, STEPS=STEPS)
            leg2_traj = self.generate_leg_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=2, t_goal=t_goal, STEPS=STEPS)
            leg4_traj = self.generate_leg_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=4, t_goal=t_goal, STEPS=STEPS)
            # Generate body trajectory
            body_traj = self.generate_body_joint_trajectory(
                step_size_xy_mult=step_size_xy_mult, DIR='N', t_goal=t_goal, STEPS=STEPS)
            # Append trajectories to q
            q = np.vstack((q,
                           np.hstack((body_traj[:, 0:7], leg0_traj[:, 7:10],
                                      body_traj[:, 10:13], leg2_traj[:, 13:16],
                                      body_traj[:, 16:19], leg4_traj[:, 19:22], body_traj[:, 22:25]))
                           ))
            # Update the current configuration
            self.qc = q[-1]
        # Log the time taken to compute the steps
        self.logger.debug(
            f'Time taken for computing {(STEP_CNT*2)+1} steps = {time()-start_time}')

        return q

    def plot_trajctory(self, state, title):
        x_figsize = 15
        y_figsize = 30
        # Create a time array based on the number of columns in x1 or x2
        time = np.arange(state.shape[0])

        # Create a single figure with a 3-row, 2-column layout
        fig, axs = plt.subplots(7, 3, figsize=(x_figsize, y_figsize))
        fig.suptitle(title)

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

        axs[1, 0].scatter(time, state[:, 3], label='x leg 0', color='blue')
        axs[1, 0].set_xlabel('time')
        axs[1, 0].set_ylabel('x  leg 0')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        axs[1, 1].scatter(time, state[:, 4], label='y  leg 0', color='blue')
        axs[1, 1].set_xlabel('time')
        axs[1, 1].set_ylabel('y  leg 0')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        axs[1, 2].scatter(time, state[:, 5], label='z  leg 0', color='blue')
        axs[1, 2].set_xlabel('time')
        axs[1, 2].set_ylabel('z  leg 0')
        axs[1, 2].legend()
        axs[1, 2].grid(True)

        axs[2, 0].scatter(time, state[:, 6], label='x  leg 1', color='blue')
        axs[2, 0].set_xlabel('time')
        axs[2, 0].set_ylabel('x leg 1')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        axs[2, 1].scatter(time, state[:, 7], label='y leg 1', color='blue')
        axs[2, 1].set_xlabel('time')
        axs[2, 1].set_ylabel('y leg 1')
        axs[2, 1].legend()
        axs[2, 1].grid(True)

        axs[2, 2].scatter(time, state[:, 8], label='z leg 1', color='blue')
        axs[2, 2].set_xlabel('time')
        axs[2, 2].set_ylabel('z leg 1')
        axs[2, 2].legend()
        axs[2, 2].grid(True)

        axs[3, 0].scatter(time, state[:, 9], label='x leg 2', color='blue')
        axs[3, 0].set_xlabel('time')
        axs[3, 0].set_ylabel('x leg 2')
        axs[3, 0].legend()
        axs[3, 0].grid(True)

        axs[3, 1].scatter(time, state[:, 10], label='y leg 2', color='blue')
        axs[3, 1].set_xlabel('time')
        axs[3, 1].set_ylabel('y leg 2')
        axs[3, 1].legend()
        axs[3, 1].grid(True)

        axs[3, 2].scatter(time, state[:, 11], label='z leg 2', color='blue')
        axs[3, 2].set_xlabel('time')
        axs[3, 2].set_ylabel('z leg 2')
        axs[3, 2].legend()
        axs[3, 2].grid(True)

        axs[4, 0].scatter(time, state[:, 12], label='x leg 3', color='blue')
        axs[4, 0].set_xlabel('time')
        axs[4, 0].set_ylabel('x leg 3')
        axs[4, 0].legend()
        axs[4, 0].grid(True)

        axs[4, 1].scatter(time, state[:, 13], label='y leg 3', color='blue')
        axs[4, 1].set_xlabel('time')
        axs[4, 1].set_ylabel('y leg 3')
        axs[4, 1].legend()
        axs[4, 1].grid(True)

        axs[4, 2].scatter(time, state[:, 14], label='z leg 3', color='blue')
        axs[4, 2].set_xlabel('time')
        axs[4, 2].set_ylabel('z leg 3')
        axs[4, 2].legend()
        axs[4, 2].grid(True)

        axs[5, 0].scatter(time, state[:, 15], label='x leg 4', color='blue')
        axs[5, 0].set_xlabel('time')
        axs[5, 0].set_ylabel('x leg 4')
        axs[5, 0].legend()
        axs[5, 0].grid(True)

        axs[5, 1].scatter(time, state[:, 16], label='y leg 4', color='blue')
        axs[5, 1].set_xlabel('time')
        axs[5, 1].set_ylabel('y leg 4')
        axs[5, 1].legend()
        axs[5, 1].grid(True)

        axs[5, 2].scatter(time, state[:, 17], label='z leg 4', color='blue')
        axs[5, 2].set_xlabel('time')
        axs[5, 2].set_ylabel('z leg 4')
        axs[5, 2].legend()
        axs[5, 2].grid(True)

        axs[6, 0].scatter(time, state[:, 18], label='x leg 5', color='blue')
        axs[6, 0].set_xlabel('time')
        axs[6, 0].set_ylabel('x leg 5')
        axs[6, 0].legend()
        axs[6, 0].grid(True)

        axs[6, 1].scatter(time, state[:, 19], label='y leg 5', color='blue')
        axs[6, 1].set_xlabel('time')
        axs[6, 1].set_ylabel('y leg 5')
        axs[6, 1].legend()
        axs[6, 1].grid(True)

        axs[6, 2].scatter(time, state[:, 20], label='z leg 5', color='blue')
        axs[6, 2].set_xlabel('time')
        axs[6, 2].set_ylabel('z leg 5')
        axs[6, 2].legend()
        axs[6, 2].grid(True)

        plt.show()


if __name__ == "__main__":
    # Create a hexapod instance with visualization and debug logging
    hexy = hexapod(init_viz=False, logging_level=logging.DEBUG)
    # Set parameters for movement
    step_size_xy_mult = 1
    # v = 0.5  # Velocity in m/s
    # t_goal = hexy.HALF_STEP_SIZE_XY / v
    t_goal = 0.02
    start_time = time()
    STEPS = 10
    # Generate trajectories for legs 0, 2, and 4
    leg0_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=0, t_goal=t_goal, STEPS=STEPS)
    leg2_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=2, t_goal=t_goal, STEPS=STEPS)
    leg4_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=4, t_goal=t_goal, STEPS=STEPS)
    # Generate body trajectory
    body_traj = hexy.generate_body_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N', t_goal=t_goal, STEPS=STEPS)
    # Combine the trajectories into a single trajectory q
    q = np.hstack((body_traj[:, 0:7], leg0_traj[:, 7:10], body_traj[:, 10:13],
                   leg2_traj[:, 13:16], body_traj[:, 16:19], leg4_traj[:, 19:22], body_traj[:, 22:25]))
    # Update the current configuration
    hexy.qc = q[-1]
    step_size_xy_mult = 2
    STEP_CNT = 3
    for i in range(0, STEP_CNT):
        # Generate trajectories for legs 1, 3, and 5
        leg1_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=1, t_goal=t_goal, STEPS=STEPS)
        leg3_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=3, t_goal=t_goal, STEPS=STEPS)
        leg5_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=5, t_goal=t_goal, STEPS=STEPS)
        # Generate body trajectory
        body_traj = hexy.generate_body_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', t_goal=t_goal, STEPS=STEPS)
        # Append trajectories to q
        q = np.vstack((q,
                       np.hstack((body_traj[:, 0:7], body_traj[:, 7:10],
                                  leg1_traj[:, 10:13], body_traj[:, 13:16],
                                  leg3_traj[:, 16:19], body_traj[:, 19:22], leg5_traj[:, 22:25]))
                       ))
        # Update the current configuration
        hexy.qc = q[-1]
        # Generate trajectories for legs 0, 2, and 4 again
        leg0_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=0, t_goal=t_goal, STEPS=STEPS)
        leg2_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=2, t_goal=t_goal, STEPS=STEPS)
        leg4_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=4, t_goal=t_goal, STEPS=STEPS)
        # Generate body trajectory
        body_traj = hexy.generate_body_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', t_goal=t_goal, STEPS=STEPS)
        # Append trajectories to q
        q = np.vstack((q,
                       np.hstack((body_traj[:, 0:7], leg0_traj[:, 7:10],
                                  body_traj[:, 10:13], leg2_traj[:, 13:16],
                                  body_traj[:, 16:19], leg4_traj[:, 19:22], body_traj[:, 22:25]))
                       ))
        # Update the current configuration
        hexy.qc = q[-1]
    # Log the time taken to compute the steps
    hexy.logger.debug(
        f'Time taken for computing {(STEP_CNT*2)+1} steps = {time()-start_time}')

    gait_angles_file_path = Path(
        f'gait_angles/gait_angles_WP{STEPS}_S{STEP_CNT}_{strftime("%Y%m%d_%H%M%S")}.npy')
    gait_angles_file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(gait_angles_file_path, q)
    # # Play the trajectory in the visualizer
    if hexy.viz_flag:
        hexy.viz.play(hexy.compute_gait())

    # hexy.plot_trajctory(title=f'Final Traj', state=np.array(
    #     [hexy.forward_kinematics(q_i) for q_i in q]))
