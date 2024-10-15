# %%
# Import necessary libraries
from time import sleep, time              # For time-related functions
import numpy as np                        # For numerical computations
import pinocchio as pin                   # For robotics computations
# For 3D geometry in MeshCat visualization
import meshcat.geometry as g
# import scipy                             # (Commented out; possibly unused)
from pathlib import Path                  # For filesystem path manipulations
import sys                                # For system-specific parameters and functions
import scipy.optimize                     # For optimization algorithms
from itertools import chain               # For chaining iterables
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# We don't want to print every decimal!
np.set_printoptions(suppress=True, precision=4)

plt.ion()


class hexapod:
    """The hexapod class represents a virtual representation of the hexapod
    that is being built as a part of our Master's Capstone Project.
    """

    def __init__(self, init_viz=True) -> None:
        """Default constructor

        Args:
            init_viz (bool, optional): Choose whether or not to initialise a MeshCat visualizer
        """
        # Load URDF directory path
        self.pin_model_dir = str(Path('URDF/URDF4Pin').absolute())
        # Load URDF file path
        self.urdf_filename = (self.pin_model_dir + '/URDF4Pin.urdf')
        # Load robot model using the URDF file
        self.robot = pin.RobotWrapper.BuildFromURDF(
            self.urdf_filename, self.pin_model_dir, root_joint=pin.JointModelFreeFlyer())
        # Initialize joint state to neutral position
        self.robot.forwardKinematics(pin.neutral(self.robot.model))
        # Update visual model placements
        self.robot.updateGeometryPlacements()
        # Set half of the step size of each foot in XY plane
        self.HALF_STEP_SIZE_XY = 0.1 / 2
        # Set the max height at which each leg can go while walking
        self.STEP_SIZE_Z = 0.025
        # Array to store frame IDs of all the feet
        self.FOOT_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "foot" in frame.name]
        # Array to store shoulder IDs of all the feet
        self.SHOULDER_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "Revolute_joint_0" in frame.name]
        # Get the frame ID of the base frame
        self.BASE_FRAME_ID = self.robot.model.getFrameId("robot_base")
        # Flag to store the state of motion
        self.state_flag = 'START'
        # qc stores the current joint state of the robot initialized to neutral
        self.qc = self.robot.q0
        # Numpy array to store the current state of the base frame and the feet frames
        self.current_state = np.concatenate((self.qc[0:7], np.concatenate(
            [pin.SE3ToXYZQUAT(self.robot.framePlacement(self.qc, frame)) for frame in self.FOOT_IDS])))
        # Numpy array to store the neutral state of the base frame and the feet frames
        self.neutral_state = np.concatenate((self.robot.q0[0:7], np.concatenate(
            [pin.SE3ToXYZQUAT(self.robot.framePlacement(self.robot.q0, frame)) for frame in self.FOOT_IDS])))

        self.BOUNDS_BASE_STAT = list(chain([(0., 0.)]*6, [(1., 1.)]))
        self.BOUNDS_BASE_MOVE_NOROT = list(
            chain([(-10., 10.)]*3, [(0., 0.)]*3, [(1., 1.)]))
        self.BOUNDS_BASE_MOVE_ROT = list(chain([(-10., 10.)]*7))
        self.BOUNDS_FOOT_STAT = [(0., 0.)]*3
        self.BOUNDS_FOOT_MOVE = [(-(np.pi/2), (np.pi/2))]*3
        # Flag to store if visualizer is needed or not
        self.viz_flag = init_viz
        if self.viz_flag == True:
            # If true, initialize the MeshCat visualizer
            self.init_viz()
        print("DEBUG POINT")

    def init_viz(self):
        """Function to initialize the visualizer and assign it to a data member"""
        try:
            # Initialize MeshCat visualizer with robot models
            self.viz = pin.visualize.MeshcatVisualizer(
                self.robot.model, self.robot.collision_model, self.robot.visual_model)
            self.viz.initViewer(open=True)
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)

        # Load the robot in the viewer
        self.viz.loadViewerModel()

        # Display coordinate frames
        self.viz.displayFrames(visibility=True)
        # Hide collision models
        self.viz.displayCollisions(visibility=False)
        # Display the robot in initial configuration
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

    def generate_state_vector(self, q):
        return np.concatenate((pin.SE3ToXYZQUAT(self.robot.framePlacement(q, self.BASE_FRAME_ID)), np.concatenate(
            [pin.SE3ToXYZQUAT(self.robot.framePlacement(q, frame)) for frame in self.FOOT_IDS])))
    # def update_current_states(self, q):
    #     """Update the current joint state and the state vector

    #     Args:
    #         q (numpy.ndarray): New joint state
    #     """
    #     self.qc = q
    #     self.current_state = np.concatenate((self.qc[0:7], np.concatenate(
    #         [self.robot.framePlacement(self.qc, frame).translation for frame in self.FOOT_IDS])))

    def state_error(self, q, desired_state=np.zeros(49)):
        """Compute the squared error between the current state and the desired state

        Args:
            q (numpy.ndarray): Current joint state
            desired_state (numpy.ndarray, optional): Desired state vector. Defaults to zeros.

        Returns:
            float: Squared error
        """
        current_state = np.concatenate((pin.SE3ToXYZQUAT(self.robot.framePlacement(q, self.BASE_FRAME_ID)), np.concatenate(
            [pin.SE3ToXYZQUAT(self.robot.framePlacement(q, frame)) for frame in self.FOOT_IDS])))
        # error = current_state - desired_state
        # return error.dot(error)
        # return np.sum((current_state - desired_state)**2)
        return np.linalg.norm(self.generate_state_vector(q) - desired_state)

    def inverse_geometery(self, q, desired_state, bounds=[(-10., 10.)]*25):
        """Inverse geometry computation to find joint states that achieve the desired state

        Args:
            q (numpy.ndarray): Starting point for minimization function (Initial Joint State)
            desired_state (numpy.ndarray): Desired state vector
            bounds (list, optional): Bounds for the joint variables. Defaults to [(-10., 10.)]*25.

        Returns:
            numpy.ndarray: Joint state corresponding to desired position
        """
        res = scipy.optimize.minimize(
            self.state_error, self.robot.q0, args=(desired_state), bounds=bounds, method='SLSQP', options={'ftol': 1e-7, 'maxiter': 1000, 'disp': True}, jac=True)
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

    def init_body_trajectory_functions(self, step_size_xy_mult, theta=0):
        """Function to initialize the linear trajectory of the robot's body

        Args:
            step_size_xy_mult (int): Multiplier to have either half or full step size
            theta (float, optional): Direction angle in radians. Defaults to 0.
        """
        # Get the direction vector based on theta
        v = self.get_direction_vector(theta=theta)[0:2]
        # Starting position of the base in XY-plane
        p1 = self.robot.framePlacement(
            self.qc, self.BASE_FRAME_ID).translation[0:2]
        # Ending position of the base in XY-plane
        p2 = p1 + (self.HALF_STEP_SIZE_XY * step_size_xy_mult * v)
        # Define the parametric functions for x(t) and y(t)
        self.x_t = lambda t: ((1 - t) * p1[0]) + (t * p2[0])
        self.y_t = lambda t:  ((1 - t) * p1[1]) + (t * p2[1])
        # TODO: Implement function for straight line interpolation of quaternion as well.
        # For now, it is assumed that the body will not rotate.

    def generate_joint_waypoints_push(self, step_size_xy_mult=1, STEPS=5, theta=0, FEET_GRP='135'):
        """Generate joint waypoints for pushing phase where the body moves forward

        Args:
            step_size_xy_mult (int, optional): Step size multiplier. Defaults to 1.
            STEPS (int, optional): Number of waypoints. Defaults to 5.
            theta (float, optional): Direction angle in radians. Defaults to 0.
            FEET_GRP (str, optional): Feet group to keep stationary. Defaults to '135'.

        Returns:
            list: List of joint configurations
        """
        # Initialize body trajectory functions
        self.init_body_trajectory_functions(
            step_size_xy_mult=step_size_xy_mult, theta=theta)
        # Generate parameter t over the range [0,1]
        s = np.linspace(0, 1, STEPS)
        # Generate waypoints for the body movement
        waypoints = np.round([list(chain([self.x_t(t), self.y_t(t), self.qc[2]], [
                             0, 0, 0, 1], self.current_state[7:])) for t in s], 5)

        # self.plot_robot_trajectory(
        #     name=f'generate_joint_waypoints_push before IK legs{FEET_GRP}', space_flag='ss', states=waypoints)

        # Define bounds for optimization based on which feet are moving
        bounds = list(chain([(-10., 10.)]*7, [(0., 0.)]*3, [(-(np.pi/2), (np.pi/2))]*3, [(0., 0.)]*3, [(-(np.pi/2), (np.pi/2))]*3, [
                      (0., 0.)]*3, [(-(np.pi/2), (np.pi/2))]*3)) if FEET_GRP == '135' else list(chain([(-10., 10.)]*7, [(-(np.pi/2), (np.pi/2))]*3, [(0., 0.)]*3, [(-(np.pi/2), (np.pi/2))]*3, [(0., 0.)]*3, [(-(np.pi/2), (np.pi/2))]*3, [(0., 0.)]*3))

        # Compute joint configurations for each waypoint
        q = [self.inverse_geometery(self.qc, wp, bounds)
             for wp in waypoints]

        self.plot_robot_trajectory(
            "generate_joint_waypoints_push after IK legs{FEET_GRP}", space_flag='js', states=q)
        return q

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
            # Calculate the index in the state vector where this foot's position is stored
            idx = int(7 + (LEG*7))
            # Initialise waypoints for the foot movement
            waypoints = np.array([self.generate_state_vector(self.qc)] * STEPS)
            # Generate waypoints for the foot movement
            for i, t in enumerate(s):
                waypoints[i, idx:idx+3] = [self.x_t(t),
                                           self.y_t(t), self.z_t(t)]
            # self.plot_robot_trajectory(
            #     name=f'generate_joint_waypoints_swing before IK leg{LEG}', space_flag='ss', states=waypoints)
            # print("DEBUG POINT")
            # Define bounds for optimization
            bounds = list(chain(self.BOUNDS_BASE_STAT, self.BOUNDS_FOOT_STAT *
                          LEG, self.BOUNDS_FOOT_MOVE,  self.BOUNDS_FOOT_STAT*(5-LEG)))
            # Compute joint configurations for each waypoint
            q[LEG] = [self.inverse_geometery(self.qc, wp, bounds)
                      for wp in waypoints]

            self.plot_robot_trajectory(
                name=f'generate_joint_waypoints_swing after IK leg{LEG}', space_flag='js', states=q[LEG])
            # Visualize the foot trajectory
            points = np.array([[self.x_t(t), self.y_t(t), self.z_t(t)]
                              for t in s]).T
            if self.viz_flag:
                self.viz.viewer[('Foot_ID_' + str(FOOT_ID) + '_trajectory')].set_object(
                    g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xffff00)))

        q = np.sum(q, axis=0)
        q[:, 6] = 1  # Ensure the quaternion component is set correctly

        self.plot_robot_trajectory(
            name=f'generate_joint_waypoints_swing after IK legs{FEET_GRP}', space_flag='js', states=q)
        return q

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

    def generate_leg_joint_trajectory(self, step_size_xy_mult, theta=0, FEET_GRP='024', STEPS=5, t_init=0, t_goal=0.1, dt=0.01):
        """Function to generate the entire trajectory of a chosen foot, from current position to goal position, in Joint state form

        Args:
            step_size_xy_mult (int): Choice between half (1) or full (2) step size
            theta (float, optional): Direction angle in radians. Defaults to 0.
            FEET_GRP (str, optional): Feet group to move. Defaults to '024'.
            STEPS (int, optional): Number of waypoints required between current and goal position. Defaults to 5.
            t_init (float, optional): Initial time. Defaults to 0.
            t_goal (float, optional): Goal time. Defaults to 0.1.
            dt (float, optional): Time divisions between t_init and t_goal. Defaults to 0.01.

        Returns:
            numpy.ndarray: A 2D array where each row forms a Joint state for the leg in its motion along the chosen path
        """
        if ((FEET_GRP != '135') & (FEET_GRP != '024')):
            raise ValueError(
                "FEET_GRP has been given an incorrect value. Accepted Values = {'024', '135'}")
        q_wps = self.generate_joint_waypoints_swing(step_size_xy_mult,
                                                    theta=theta, FEET_GRP=FEET_GRP, STEPS=STEPS)
        q_traj = self.qc
        # Create a mask to select which parts of the state vector to update
        mask = np.concatenate((np.zeros(6), [1], np.zeros(3), np.ones(3), np.zeros(3), np.ones(3), np.zeros(3), np.ones(
            3))) if FEET_GRP == '135' else np.concatenate((np.zeros(6), [1], np.ones(3), np.zeros(3), np.ones(3), np.zeros(3), np.ones(3), np.zeros(3)))
        # Generate trajectory by interpolating between waypoints
        for i in range(0, q_wps.__len__()-1):
            t = t_init
            while t < t_goal:
                q_t = self.compute_trajectory_p(
                    q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
                q_traj = np.vstack((q_traj, np.multiply(q_t, mask)))
                t = (t + dt)

        # self.plot_robot_trajectory(
        #     name='generate_joint_leg_trajectory', space_flag='js', q=q_wps)
        return np.delete(q_traj, 0, axis=0)  # Remove the initial state

    def generate_body_trajectory(self, step_size_xy_mult, theta=0, FEET_GRP='135', STEPS=5, t_init=0, t_goal=0.1, dt=0.01):
        """Generate the trajectory of the robot's body during the push phase

        Args:
            step_size_xy_mult (int): Multiplier for step size
            theta (float, optional): Direction angle in radians. Defaults to 0.
            FEET_GRP (str, optional): Feet group that remains stationary. Defaults to '135'.
            STEPS (int, optional): Number of waypoints. Defaults to 5.
            t_init (float, optional): Initial time. Defaults to 0.
            t_goal (float, optional): Goal time. Defaults to 0.1.
            dt (float, optional): Time step. Defaults to 0.01.

        Returns:
            numpy.ndarray: Trajectory of joint configurations
        """
        if ((FEET_GRP != '135') & (FEET_GRP != '024')):
            raise ValueError(
                "FEET_GRP has been given an incorrect value. Accepted Values = {'024', '135'}")
        q_wps = self.generate_joint_waypoints_push(
            step_size_xy_mult=step_size_xy_mult, STEPS=STEPS, theta=theta, FEET_GRP=FEET_GRP)
        self.plot_robot_trajectory(
            name='generate_body_trajectory', space_flag='js', states=q_wps)
        q_traj = self.qc
        # Create a mask to select which parts of the state vector to update
        mask = np.concatenate((np.ones(7), np.zeros(3), np.ones(3), np.zeros(3), np.ones(3), np.zeros(3), np.ones(
            3))) if FEET_GRP == '135' else np.concatenate((np.ones(7), np.ones(3), np.zeros(3), np.ones(3), np.zeros(3), np.ones(3), np.zeros(3)))
        # Generate trajectory by interpolating between waypoints
        for i in range(0, q_wps.__len__()-1):
            t = t_init
            while t < t_goal:
                q_t = self.compute_trajectory_p(
                    q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
                q_traj = np.vstack((q_traj, np.multiply(q_t, mask)))
                t = (t + dt)
        return np.delete(q_traj, 0, axis=0)  # Remove the initial state

    def body_velocity_error(self, q, desired_pos=np.zeros(3)):
        """Placeholder for body velocity error computation

        Args:
            q (numpy.ndarray): Joint state
            desired_pos (numpy.ndarray, optional): Desired position. Defaults to zeros.
        """
        pass

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
        plt.show()


# %%
if __name__ == "__main__":
    # Instantiate the hexapod with visualization enabled
    hexy = hexapod(init_viz=True)
    vel = 2  # Velocity
    # Compute goal time based on step size and velocity
    t_goal = (hexy.HALF_STEP_SIZE_XY * 2) / vel
    sleep(3)  # Wait for 3 seconds
    theta = 45  # Direction angle in radians
    step_size_xy_mult = 1  # Step size multiplier
    STEPS = 5
    # Generate leg and body trajectories
    leg024_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, theta=theta, FEET_GRP='024', t_goal=t_goal, STEPS=STEPS)
    # leg135_traj = hexy.generate_body_trajectory(
    #     step_size_xy_mult=step_size_xy_mult, theta=theta, FEET_GRP='135', t_goal=t_goal, STEPS=STEPS)
    # legs_traj = leg024_traj + leg135_traj  # Combine trajectories
    # hexy.plot_robot_trajectory(name='legs_traj', q=legs_traj, space_flag='js')
    start_time = time()
    for q in leg024_traj:
        # Display the current joint configuration
        hexy.viz.display(q)
        # hexy.update_current_states(q)  # Update the current state

    print(f"Time taken for this step = {time() - start_time}")

    # step_size_xy_mult = 2
    # while True:
    #     # Generate body trajectory for feet group '024' and leg trajectory for '135'
    #     leg024_traj = hexy.generate_body_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, theta=theta, FEET_GRP='024', t_goal=t_goal)
    #     leg135_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, theta=theta, FEET_GRP='135', t_goal=t_goal)
    #     legs_traj = leg024_traj + leg135_traj
    #     start_time = time()
    #     for q in legs_traj:
    #         hexy.viz.display(q)
    #         hexy.update_current_states(q)
    #     print(f"Time taken for this step = {time() - start_time}")

    #     # Generate leg trajectory for feet group '024' and body trajectory for '135'
    #     leg024_traj = hexy.generate_leg_joint_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, theta=theta, FEET_GRP='024', t_goal=t_goal)
    #     leg135_traj = hexy.generate_body_trajectory(
    #         step_size_xy_mult=step_size_xy_mult, theta=theta, FEET_GRP='135', t_goal=t_goal)
    #     legs_traj = leg024_traj + leg135_traj
    #     start_time = time()
    #     for q in legs_traj:
    #         hexy.viz.display(q)
    #         hexy.update_current_states(q)
    #     print(f"Time taken for this step = {time() - start_time}")

    # step_size_xy_mult = 2
    # leg0_traj = hexy.generate_leg_joint_trajectory(
    #     step_size_xy_mult=step_size_xy_mult, DIR='S', LEG=0)
    # leg2_traj = hexy.generate_leg_joint_trajectory(
    #     step_size_xy_mult=step_size_xy_mult, DIR='S', LEG=2)
    # leg4_traj = hexy.generate_leg_joint_trajectory(
    #     step_size_xy_mult=step_size_xy_mult, DIR='S', LEG=4)
    # body_traj = hexy.generate_body_trajectory(
    #     step_size_xy_mult=step_size_xy_mult, DIR='S', FEET_GRP='135')
    # leg024_traj = leg0_traj + leg2_traj + leg4_traj
    # # leg024_traj[:, 6] = 1
    # start_time = time()
    # for q in leg024_traj:
    #     hexy.viz.display(q)
    #     hexy.qc = q
    # print(f"Time taken for this step = {time() - start_time}")
