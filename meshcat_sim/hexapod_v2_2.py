from time import sleep, time, strftime
import numpy as np
import pinocchio as pin
import meshcat.geometry as g
from pathlib import Path
from sys import exit
from scipy.optimize import minimize
from itertools import chain
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import logging

# we don't want to print every decimal!
np.set_printoptions(suppress=True, precision=5)


class hexapod:

    def __init__(self, init_viz=True, show_plots=True, save_plots=True, logging_level=logging.DEBUG) -> None:
        self.init_logger(logging_level=logging_level)
        self.pin_model_dir = str(
            Path('./URDF/URDF4Pin').absolute())
        self.urdf_filename = (self.pin_model_dir + '/URDF4Pin.urdf')

        self.robot = pin.RobotWrapper.BuildFromURDF(
            self.urdf_filename, self.pin_model_dir, root_joint=pin.JointModelFreeFlyer())

        self.show_plots = show_plots
        self.save_plots = save_plots

        self.robot.forwardKinematics(pin.neutral(self.robot.model))
        self.robot.updateGeometryPlacements()
        self.direction_slope = 0.0
        self.HALF_STEP_SIZE_XY = 0.06/2
        self.STEP_SIZE_Z = 0.025

        # Array to store frame IDs of all the feet
        self.FOOT_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "foot" in frame.name]
        # Get the frame ID of the base frame
        self.BASE_FRAME_ID = self.robot.model.getFrameId("robot_base")
        self.state_flag = 'START'
        self.qc = self.robot.q0
        self.viz_flag = init_viz
        # Numpy array to store the neutral state of the base frame and the feet frames
        self.neutral_state = np.concatenate((self.robot.q0[0:7], np.concatenate([pin.SE3ToXYZQUAT(self.robot.framePlacement(
            self.robot.q0, frame_id)) for frame_id in self.FOOT_IDS])))

        self.BOUNDS_BASE_STAT = list(chain([(0., 0.)]*6, [(1., 1.)]))
        self.BOUNDS_BASE_MOVE_NOROT = list(
            chain([(-10., 10.)]*3, [(0., 0.)]*3, [(1., 1.)]))
        self.BOUNDS_BASE_MOVE_ROT = list(chain([(-10., 10.)]*7))
        self.BOUNDS_FOOT_STAT = [(0., 0.)]*3
        self.BOUNDS_FOOT_MOVE = [(-(np.pi/2), (np.pi/2))]*3
        if self.viz_flag == True:
            self.init_viz()
        logging.debug(
            f"Hexapod Object Initialised Successfully with init_viz = {self.viz_flag}, show_plots={self.show_plots}, save_plots={self.save_plots}")

    def init_logger(self, logging_level):
        # Define the log directory
        log_dir = Path('logs').absolute()
        log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for log file
        timestamp = strftime("%Y%m%d_%H%M%S")
        log_file = f'hexapod_log_{timestamp}.log'
        log_path = log_dir / log_file

        # Create logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)
        self.logger.propagate = False

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Filters
        excluded_loggers = ['some_imported_module', 'another_module']

        class ExcludeLoggersFilter(logging.Filter):
            def filter(self, record):
                return record.name not in excluded_loggers

        exclude_filter = ExcludeLoggersFilter()
        file_handler.addFilter(exclude_filter)
        console_handler.addFilter(exclude_filter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def init_viz(self):
        try:
            self.viz = pin.visualize.MeshcatVisualizer(
                self.robot.model, self.robot.collision_model, self.robot.visual_model)
            self.viz.initViewer(open=True)
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            exit(0)

        # Load the robot in the viewer.
        self.viz.loadViewerModel()

        self.viz.displayFrames(visibility=True)
        self.viz.displayCollisions(visibility=False)
        self.viz.display(self.qc)

    def print_all_frame_info(self):
        """Print all the frame names, IDs, and positions"""
        for frame in self.robot.model.frames:
            # Print the frame name, type, ID, and position
            print(
                f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")

    def find_frame_info_by_name(self, name: str):
        """Print only those frames whose names contain a matching string

        Args:
            name (str): Name matching string
        """
        for frame in self.robot.model.frames:
            if name in frame.name:
                # Print the frame name, type, ID, and position
                print(
                    f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")

    def rotateZ(self, theta):
        """Create a rotation matrix around the Z-axis by angle theta

        Args:
            theta (float): Rotation angle in radians

        Returns:
            numpy.ndarray: Rotation matrix
        """
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    # Function to calculate the skew-symmetric matrix of a vector

    def skew_symmetric(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    # Function to compute the orientation error e_O based on Equation (3.87)

    def orientation_error(self, Q_d, Q):
        Q_d = pin.Quaternion(np.array(Q_d)).normalized().coeffs()
        Q = pin.Quaternion(np.array(Q)).normalized().coeffs()

        # Extract scalar and vector parts from the quaternions (in [x, y, z, w] format)
        epsilon_d = np.array(Q_d[:3])  # Vector part of Q_d (x, y, z)
        eta_d = Q_d[3]                 # Scalar part of Q_d (w)

        epsilon_q = np.array(Q[:3])    # Vector part of Q (x, y, z)
        eta_q = Q[3]                   # Scalar part of Q (w)

        # Compute the skew-symmetric matrix of epsilon_d
        S_eps_d = self.skew_symmetric(epsilon_d)

        # Compute the orientation error
        e_O = eta_q * epsilon_d - eta_d * epsilon_q - S_eps_d.dot(epsilon_q)

        return np.linalg.norm(e_O)

    def quaternion_product(self, q1=np.array([0., 0., 0., 1.]), q2=np.array([0., 0., 0., 1.])):
        """
        Compute the product of two quaternions.

        Args:
            q1 (numpy.ndarray): First quaternion [x, y, z, w].
            q2 (numpy.ndarray): Second quaternion [x, y, z, w].

        Returns:
            numpy.ndarray: Product quaternion [x, y, z, w].
        """
        eta_1, eta_2 = q1[3], q2[3]
        eps_1, eps_2 = q1[:3], q2[:3]

        eps = eta_1 * eps_2 + eta_2 * eps_1 + np.cross(eps_1, eps_2)
        eta = eta_1 * eta_2 - np.dot(eps_1, eps_2)

        return np.concatenate((eps, [eta]))

    def get_direction_vector(self, theta):
        """Get the direction vector based on angle theta

        Args:
            theta (float): Direction angle in radians

        Returns:
            numpy.ndarray: Direction vector
        """
        # Rotate the unit vector along the X-axis by theta around the Z-axis
        v = self.rotateZ(theta=theta) @ np.array([1, 0, 0])
        if self.viz_flag:
            # Visualize the direction vector in MeshCat
            self.viz.viewer['direction_vector'].set_object(
                g.Line(g.PointsGeometry(np.array(([0, 0, 0], v)).T
                                        ), g.MeshBasicMaterial(color=0xffff00)))
        return v

    def frame_pose_err(self, q, FRAME_ID=9, desired_pose=np.concatenate((np.zeros((6, 1)), np.ones((1, 1))))):
        current_pose = pin.SE3ToXYZQUAT(
            self.robot.framePlacement(q=q, index=FRAME_ID))
        error_pos_norm = np.linalg.norm(current_pose[0:3] - desired_pose[0:3])
        error_quat_norm = self.orientation_error(
            Q_d=desired_pose[3:7], Q=current_pose[3:7])
        return ((error_pos_norm**2) + (error_quat_norm**2))

    def inverse_geometry(self, q, bounds, FRAME_ID=2, desired_pose=np.concatenate((np.zeros((6, 1)), np.ones((1, 1))))):
        res = minimize(
            self.frame_pose_err, q, args=(FRAME_ID, desired_pose), bounds=bounds, method='SLSQP', options={'ftol': 1e-7, 'maxiter': 1000, 'disp': True})
        return res.x

    def init_foot_trajectory_functions(self, step_size_xy_mult, theta=0, FOOT_ID=10):
        """Function to initialize the parabolic trajectory of a foot

        Args:
            step_size_xy_mult (int): Multiplier to have either half or full step size
            theta (float, optional): Direction angle in radians. Defaults to 0.
            FOOT_ID (int, optional): Chosen foot frame ID. Defaults to 10.
        """
        # Get the direction vector based on theta
        v = self.get_direction_vector(theta=theta)[0:2]
        # Starting position of the foot in XY-plane
        p1 = self.robot.framePlacement(self.qc, FOOT_ID).translation[0:2]
        # Ending position of the foot in XY-plane
        p2 = p1 + (self.HALF_STEP_SIZE_XY * step_size_xy_mult * v)
        # Define the parametric functions for x(t) and y(t)
        self.x_t = lambda t: ((1 - t) * p1[0]) + (t * p2[0])
        self.y_t = lambda t:  ((1 - t) * p1[1]) + (t * p2[1])

        # Define the parabolic function for z(t)
        self.z_t = lambda t: (((-self.STEP_SIZE_Z)/0.25) *
                              (t ** 2)) + ((self.STEP_SIZE_Z / 0.25) * t)

    def generate_joint_waypoints_swing(self, step_size_xy_mult, STEPS=5, theta=0, FEET_GRP='024'):
        """Function to generate waypoints along the parabolic trajectory of the foot

        Args:
            step_size_xy_mult (int): Multiplier to have either half or full step size
            STEPS (int, optional): Number of waypoints along the trajectory. Defaults to 5.
            theta (float, optional): Direction angle in radians. Defaults to 0.
            FEET_GRP (str, optional): Feet group to move. Defaults to '024'.

        Returns:
            numpy.ndarray: Array of joint spaces, where each element corresponds to a computed waypoint along the parabolic trajectory of the foot
        """
        FEET = [int(char) for char in list(FEET_GRP)]
        # Initialize an array to store joint configurations
        q = np.zeros((6, STEPS, self.robot.q0.size))
        for LEG in FEET:
            FOOT_ID = self.FOOT_IDS[LEG]
            # Initialize foot trajectory functions for the specific foot
            self.init_foot_trajectory_functions(
                step_size_xy_mult=step_size_xy_mult, theta=theta, FOOT_ID=FOOT_ID)
            s = np.linspace(0, 1, STEPS)
            # Generate waypoints for the foot movement
            waypoints = [[round(self.x_t(t), 5), round(
                self.y_t(t), 5), round(self.z_t(t), 5), 0, 0, 0, 1] for t in s]
            if LEG == 0:
                waypoints_plt = np.array([np.concatenate((self.neutral_state[0:7], [round(self.x_t(t), 5), round(
                    self.y_t(t), 5), round(self.z_t(t), 5), 0, 0, 0, 1], self.neutral_state[14:])) for t in s])
            else:
                waypoints_plt = np.array([np.concatenate((self.neutral_state[0:7], self.neutral_state[7:(LEG+1)*7], [round(self.x_t(t), 5), round(
                    self.y_t(t), 5), round(self.z_t(t), 5), 0, 0, 0, 1], self.neutral_state[((LEG+1)*7)+7:])) for t in s])
            self.plot_robot_trajectory(
                name=f'generate_joint_waypoints_swing before IK leg{LEG}', space_flag='ss', states=waypoints_plt)
            # Define bounds for optimization
            bounds = list(chain(self.BOUNDS_BASE_STAT, self.BOUNDS_FOOT_STAT *
                          LEG, self.BOUNDS_FOOT_MOVE,  self.BOUNDS_FOOT_STAT*(5-LEG)))
            # Compute joint configurations for each waypoint
            q[LEG] = [self.inverse_geometry(q=self.qc, desired_pose=wp, bounds=bounds, FRAME_ID=FOOT_ID)
                      for wp in waypoints]
            # Visualize the foot trajectory
            points = np.array([waypoints[0], waypoints[4]]).T
            if self.viz_flag:
                self.viz.viewer[('Foot_ID_' + str(FOOT_ID) + '_trajectory')].set_object(
                    g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xffff00)))

        q = np.sum(q, axis=0)
        q[:, 6] = 1  # Ensure the quaternion component is set correctly

        self.plot_robot_trajectory(
            name=f'generate_joint_waypoints_swing after IK legs{FEET_GRP}', space_flag='js', states=q)
        return q

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

    def generate_state_vector(self, q):
        return np.concatenate((pin.SE3ToXYZQUAT(self.robot.framePlacement(q, self.BASE_FRAME_ID)), np.concatenate(
            [pin.SE3ToXYZQUAT(self.robot.framePlacement(q, frame)) for frame in self.FOOT_IDS])))

    def plot_robot_trajectory(self, name: str, states=None, space_flag='js'):
        """
        Plots the robot's body position, orientation, and foot positions in separate 3D subplots for multiple states.

        Parameters:
            states (np.array): A numpy array of shape (N, 25) where:
                - states[:, 0:3] represents the positions of the robot body.
                - states[:, 3:7] represents the quaternions of the robot body.
                - states[:, 7:10] represents the positions of foot 0.
                - states[:, 10:13] represents the positions of foot 1.
                - states[:, 13:16] represents the positions of foot 2.
                - states[:, 16:19] represents the positions of foot 3.
                - states[:, 19:22] represents the positions of foot 4.
                - states[:, 22:25] represents the positions of foot 5.
        """

        if ((space_flag != 'js') & (space_flag != 'ss')):
            raise ValueError(
                "space_flag has been given an incorrect value. Accepted Values = {'js', 'ss'}")

        if space_flag == 'js':
            q = states
            states = np.array([self.generate_state_vector(qi) for qi in q])

        fig = plt.figure(name, figsize=(16, 12))

        # Body position plot
        ax1 = fig.add_subplot(241, projection='3d')
        body_positions = states[:, 0:3]
        ax1.plot(body_positions[:, 0], body_positions[:, 1],
                 body_positions[:, 2], 'r-', label='Body Position')
        ax1.set_title('Body Position')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Body orientation plot (plot each axis separately as a 3D line)
        ax2 = fig.add_subplot(242, projection='3d')
        ax2.set_title('Body Orientation (quiver at first and last state)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        body_axes_length = 0.1
        rotation_first = R.from_quat(states[0, 3:7]).as_matrix()
        rotation_last = R.from_quat(states[-1, 3:7]).as_matrix()
        for i, color in enumerate(['r', 'g', 'b']):  # X, Y, Z axes
            ax2.quiver(body_positions[0, 0], body_positions[0, 1], body_positions[0, 2],
                       rotation_first[0, i], rotation_first[1,
                                                            i], rotation_first[2, i],
                       color=color, length=body_axes_length, label=f'Start' if i == 0 else "")
            ax2.quiver(body_positions[-1, 0], body_positions[-1, 1], body_positions[-1, 2],
                       rotation_last[0, i], rotation_last[1,
                                                          i], rotation_last[2, i],
                       color=color, length=body_axes_length, linestyle='dotted', label=f'End' if i == 0 else "")
        ax2.legend()

        # Foot trajectories (one plot per foot)
        for i, foot_index in enumerate(range(7, 49, 7)):
            ax = fig.add_subplot(2, 4, i + 3, projection='3d')
            foot_positions = states[:, foot_index:foot_index+3]
            ax.plot(foot_positions[:, 0], foot_positions[:, 1],
                    foot_positions[:, 2], label=f'Foot {i} Position')
            # ax.set_title(f'Foot {i} Position')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()

        plt.tight_layout()
        if self.save_plots:
            plt.savefig(
                f'/media/Shantanu/hexapod/meshcat_sim/plots/{name}_{strftime("%Y%m%d_%H%M%S")}.png', dpi=600)
        if self.show_plots:
            plt.show()


if __name__ == '__main__':

    hexy = hexapod(init_viz=False, save_plots=False,
                   show_plots=False, logging_level=logging.DEBUG)
    # hexy.generate_joint_waypoints_swing(step_size_xy_mult=1)
