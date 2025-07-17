# we import useful libraries
from time import sleep, time
import numpy as np
import pinocchio as pin
import meshcat.geometry as g
# import scipy
from pathlib import Path
import sys
import scipy.optimize

# we don't want to print every decimal!
np.set_printoptions(suppress=True, precision=4)


class hexapod:
    """The hexapod class represents a virtual representation of the hexapod that is being built as a part of our Master's Capstone Project.
    """

    def __init__(self, init_viz=True) -> None:
        """Default constructor

        Args:
            init_viz (bool, optional): Choose whether or not to initialise a meshcat visualiser
        """
        # load URDF directory path
        self.pin_model_dir = str(Path('./meshcat_sim/URDF/URDF4Pin').absolute())
        # load URDF file path
        self.urdf_filename = (self.pin_model_dir + '/URDF4Pin.urdf')
        # load robot model
        self.robot = pin.RobotWrapper.BuildFromURDF(
            self.urdf_filename, self.pin_model_dir, root_joint=pin.JointModelFreeFlyer())
        # initialise joint state
        self.robot.forwardKinematics(pin.neutral(self.robot.model))
        # update visual_model placements
        self.robot.updateGeometryPlacements()
        # set half of the step size of each foot in XY plane
        self.HALF_STEP_SIZE_XY = 0.06/2
        # set the max height at which each leg can go while walking
        self.STEP_SIZE_Z = 0.025
        # array to store joint IDs of all the feet
        self.FOOT_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "foot" in frame.name]
        # array to store shoulder IDs of all the feet
        self.SHOULDER_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "Revolute_joint_0" in frame.name]
        self.BASE_FRAME_ID = self.robot.model.getFrameId("robot_base")
        # Flag to store the state of motion
        self.state_flag = 'START'
        # qc stores the current joint state of the robot to neutral
        self.qc = self.robot.q0
        # NP array to store the current state of the base frame and the feet frames
        self.current_state = np.concatenate((self.qc[0:7], np.concatenate(
            [self.robot.framePlacement(self.qc, frame).translation for frame in self.FOOT_IDS])))
        # flag to store if visualiser is needed or not
        self.viz_flag = init_viz
        if self.viz_flag == True:
            # if true, initialise the meshcat visualiser
            self.init_viz()
        print("DEBUG POINT")

    def init_viz(self):
        """Function to initialise the visualiser and assign it to a data member
        """
        try:
            self.viz = pin.visualize.MeshcatVisualizer(
                self.robot.model, self.robot.collision_model, self.robot.visual_model)
            self.viz.initViewer(open=True)
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)

        # Load the robot in the viewer.
        self.viz.loadViewerModel()

        self.viz.displayFrames(visibility=True)
        self.viz.displayCollisions(visibility=False)
        self.viz.display(self.qc)

    def state_error(self, q, desired_state=np.zeros(25)):
        current_state = np.concatenate((pin.SE3ToXYZQUAT(self.robot.framePlacement(q, self.BASE_FRAME_ID)), np.concatenate(
            [self.robot.framePlacement(q, frame).translation for frame in self.FOOT_IDS])))
        error = current_state - desired_state
        return error.dot(error)

    def foot_pos_err(self, q, FRAME_ID=10, desired_pos=np.zeros(3)):
        """Function to obtain the error in the foot position (i.e. (current - desired)**2)

        Args:
            q (_type_): Joint state matrix
            FRAME_ID (int, optional): Foot ID of chosen foot. Defaults to 10.
            desired_pos (_type_, optional): Desired position of chosen foot. Defaults to np.zeros(3).

        Returns:
            _type_: error in foot position, squared
        """
        self.robot.forwardKinematics(q)
        current_pos = self.robot.framePlacement(q, FRAME_ID).translation
        error = current_pos - desired_pos
        return error.dot(error)

    def print_all_frame_info(self):
        """Print all the frame names, IDs and positions
        """
        for frame in self.robot.model.frames:
            # Print the frame name and type
            print(
                f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")

    def find_frame_info_by_name(self, name: str):
        """Print only those frames whose names contain matching string

        Args:
            name (str): Name matching string
        """
        for frame in self.robot.model.frames:
            if name in frame.name:
                print(
                    f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")

    def inverse_geometery(self, q, FRAME_ID, desired_pos, bounds=[(-10., 10.)]*25):
        """Inverse geometry computation

        Args:
            q (_type_): Starting point for minimisation function (Initial Joint State)
            FRAME_ID (int, optional): End effector frame ID. Defaults to 10.
            desired_pos (_type_, optional): Desired position of end effector. Defaults to np.zeros(3).

        Returns:
            _type_: Joint state corresponding to desired position
        """
        # res = scipy.optimize.minimize(
        #     self.foot_pos_err, q, args=(FRAME_ID, desired_pos), bounds=bounds)
        desired_state = np.concatenate((self.robot.q0[0:7], np.concatenate(
            [self.robot.framePlacement(self.robot.q0, frame).translation for frame in self.FOOT_IDS])))
        if FRAME_ID == 2:
            desired_state[0:7] = desired_pos
        else:
            i = ((int(np.ceil(FRAME_ID/10)) - 1) * 3) + 7
            desired_state[i:(i+3)] = desired_pos

        res = scipy.optimize.minimize(
            self.state_error, q, args=(desired_state), bounds=bounds)
        return res.x

    def north_vector(self):
        """Computes the vector pointing North of the robot

        Returns:
            _type_: Vector pointing North
        """
        # Get frame of Joint_01
        p1 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[0]).translation
        # Get frame for Joint_02
        p2 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[1]).translation
        # find vector
        v = np.array([(p2 - p1)[1], -(p2 - p1)[0]])
        v = v/(np.sqrt((v[0]**2) + (v[1]**2)))
        if self.viz_flag:
            self.viz.viewer['direction_vector'].set_object(
                g.Line(g.PointsGeometry(np.array(([0, 0, 0], np.hstack((v, [0.0])))).T
                                        ), g.MeshBasicMaterial(color=0xffff00)))
        return v

    def south_vector(self):
        """Computes the vector pointing South of the robot

        Returns:
            _type_: Vector pointing South
        """
        # Get frame of Joint_01
        p1 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[3]).translation
        # Get frame for Joint_02
        p2 = self.robot.framePlacement(
            self.qc, self.SHOULDER_IDS[4]).translation
        # find vector
        v = np.array([(p2 - p1)[1], -(p2 - p1)[0]])
        v = v/(np.sqrt((v[0]**2) + (v[1]**2)))
        if self.viz_flag:
            self.viz.viewer['direction_vector'].set_object(
                g.Line(g.PointsGeometry(np.array(([0, 0, 0], np.hstack((v, [0.0])))).T
                                        ), g.MeshBasicMaterial(color=0xffff00)))
        return v

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
        """Generates the vector pointing in the chosen direction

        Args:
            DIR (str, optional): Chosen direction. Accepted values {N, S, E, W, NE, NW, SE, SW}. Defaults to 'N'.

        Returns:
            _type_: 2D vector in XY plane that points in the chosen direction
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
        """Function to initialise the parabolic trajectory of a foot

        Args:
            step_size_xy_mult (_type_): Multiplier to have either half or full step size
            DIR (str, optional): Chosen direction of motion. Defaults to 'N'
            FOOT_ID (int, optional): Chosen foot frame ID. Defaults to 10
        """
        v = self.generate_direction_vector(DIR=DIR)
        p1 = self.robot.framePlacement(self.qc, FOOT_ID).translation[0:2]
        p2 = p1 + (self.HALF_STEP_SIZE_XY * step_size_xy_mult * v)
        self.x_t = lambda t: ((1 - t) * p1[0]) + (t * p2[0])
        self.y_t = lambda t:  ((1 - t) * p1[1]) + (t * p2[1])

        self.z_t = lambda t: (((-self.STEP_SIZE_Z)/0.25) *
                              (t ** 2)) + ((self.STEP_SIZE_Z / 0.25) * t)

    def init_body_trajectory_functions(self, step_size_xy_mult, DIR='N'):
        """Function to initialise the parabolic trajectory of a foot

        Args:
            step_size_xy_mult (_type_): Multiplier to have either half or full step size
            DIR (str, optional): Chosen direction of motion. Defaults to 'N'
            FOOT_ID (int, optional): Chosen foot frame ID. Defaults to 10
        """
        v = self.generate_direction_vector(DIR=DIR)
        p1 = self.robot.framePlacement(
            self.qc, self.BASE_FRAME_ID).translation[0:2]
        p2 = p1 + (self.HALF_STEP_SIZE_XY * step_size_xy_mult * v)
        self.x_t = lambda t: ((1 - t) * p1[0]) + (t * p2[0])
        self.y_t = lambda t:  ((1 - t) * p1[1]) + (t * p2[1])
        # TODO: Implement function for straight line interpolation of quaternion as well. For now, it shall be assumed that the body will not rotate
        # self.z_t = lambda t: (((-self.STEP_SIZE_Z)/0.25) *
        #                       (t ** 2)) + ((self.STEP_SIZE_Z / 0.25) * t)

    def generate_joint_waypoints_push(self, step_size_xy_mult, STEPS=5, DIR='N', FEET_GRP='135'):
        self.init_body_trajectory_functions(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR)
        s = np.linspace(0, 1, STEPS)
        waypoints = [[round(self.x_t(t), 5), round(
            self.y_t(t), 5), round(self.qc[2], 5), 0, 0, 0, 1] for t in s]
        # points = np.array([waypoints[0], waypoints[4]]).T
        bounds = [(-10., 10.)]*25
        if FEET_GRP == '135':
            bounds[7:10] = [(0., 0.)]*3
            bounds[13:16] = [(0., 0.)]*3
            bounds[19:22] = [(0., 0.)]*3
        else:
            bounds[10:13] = [(0., 0.)]*3
            bounds[16:19] = [(0., 0.)]*3
            bounds[22:25] = [(0., 0.)]*3

        return [self.inverse_geometery(self.qc, self.BASE_FRAME_ID, wp, bounds)
                for wp in waypoints]

    def generate_joint_waypoints_swing(self, step_size_xy_mult, STEPS=5, DIR='N', FOOT_ID=10):
        """Function to generate waypoints along the generated parabolic trajectory of the foot

        Args:
            step_size_xy_mult (_type_): Multiplier to have either half or full step size
            STEPS (int, optional): Number of waypoints along the trajectory. Defaults to 5.
            DIR (str, optional): Chosen direction of motion. Defaults to 'N'
            FOOT_ID (int, optional): Chosen foot frame ID. Defaults to 10

        Returns:
            _type_: Array of joint spaces, where each element corresponds to a computed waypoint along the parabolic trajectory of the foot
        """
        self.init_foot_trajectory_functions(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, FOOT_ID=FOOT_ID)
        s = np.linspace(0, 1, STEPS)
        waypoints = [[round(self.x_t(t), 5), round(
            self.y_t(t), 5), round(self.z_t(t), 5)] for t in s]
        points = np.array([waypoints[0], waypoints[4]]).T

        if self.viz_flag:
            self.viz.viewer[('Foot_ID_' + str(FOOT_ID) + '_trajectory')].set_object(
                g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xffff00)))
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
        return [self.inverse_geometery(self.qc, FOOT_ID, wp, bounds)
                for wp in waypoints]

    def compute_trajectory(self, position_init, position_goal, t_init, t_goal, t):
        """Time parameterisation of the trajectory with acceleation and velocity constraints 

        Args:
            position_init (_type_): Initial position in Joint state form
            position_goal (_type_): Goal position in Joint state form
            t_init (_type_): Initial time
            t_goal (_type_): Goal time
            t (_type_): Current time

        Returns:
            _type_: For time t : desired position in Joint state, desired velocity as a differential of Joint state, desired acceleration as a double differential of Joint state
        """
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

        return self.desired_position, self.desired_velocity, self.desired_acceleration

    def generate_leg_joint_trajectory(self, step_size_xy_mult, DIR='N', LEG=0, STEPS=5, t_init=0, t_goal=0.1, dt=0.01):
        """Fucntion to generate the entire trajectory of a chosen foot, from current position to goal position, in Joint state form

        Args:
            step_size_xy_mult (_type_): Choice between half (1) or full (2) step size. Possible vales {1, 2}
            DIR (str, optional): Chosen direction of motion. Defaults to 'N'.
            LEG (int, optional): Chosen leg. Defaults to 0.
            STEPS (int, optional): Number of waypoints required between current and goal postion. Defaults to 5.
            t_init (int, optional): Initial time. Defaults to 0.
            t_goal (float, optional): Goal time. Defaults to 0.1.
            dt (float, optional): Time divisions between t_init and t_goal. Defaults to 0.01.

        Returns:
            _type_: A 2D array, where each row forms a Joint state for the leg in its motion along the chosen path
        """
        q_wps = self.generate_joint_waypoints_swing(step_size_xy_mult,
                                                    DIR=DIR, FOOT_ID=self.FOOT_IDS[LEG], STEPS=STEPS)
        q_traj = self.qc
        mask = np.concatenate((np.zeros(7), np.zeros(
            LEG*3), [1, 1, 1], np.zeros((5-LEG)*3)))
        for i in range(0, q_wps.__len__()-1):
            t = t_init
            while t < t_goal:
                q_t = self.compute_trajectory(
                    q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
                q_traj = np.vstack((q_traj, np.multiply(q_t, mask)))
                t = (t + dt)
        return np.delete(q_traj, 0, axis=0)

    def generate_body_trajectory(self, step_size_xy_mult, DIR='N', FEET_GRP='135', STEPS=5, t_init=0, t_goal=0.1, dt=0.01):
        if ((FEET_GRP != '135') & (FEET_GRP != '024')):
            raise ValueError(
                "FEET_GRP has been given an incorrect value. Accepted Values = {'024', '135'}")
        q_wps = self.generate_joint_waypoints_push(
            step_size_xy_mult=step_size_xy_mult, STEPS=STEPS, DIR=DIR, FEET_GRP=FEET_GRP)
        q_traj = self.qc
        mask = np.concatenate((np.ones(7), np.zeros(3), np.ones(3), np.zeros(3), np.ones(3), np.zeros(3), np.ones(
            3))) if FEET_GRP == '135' else np.concatenate((np.ones(10), np.zeros(3), np.ones(3), np.zeros(3), np.ones(3), np.zeros(3)))
        for i in range(0, q_wps.__len__()-1):
            t = t_init
            while t < t_goal:
                q_t = self.compute_trajectory(
                    q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
                q_traj = np.vstack((q_traj, np.multiply(q_t, mask)))
                t = (t + dt)
        return np.delete(q_traj, 0, axis=0)

    def body_velocity_error(self, q, desired_pos=np.zeros(3)):
        pass


if __name__ == "__main__":
    hexy = hexapod(init_viz=True)
    sleep(3)
    step_size_xy_mult = 1
    leg0_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=0, t_goal=0.05)
    leg2_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=2, t_goal=0.05)
    leg4_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=4, t_goal=0.05)
    body_traj = hexy.generate_body_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N', FEET_GRP='135', t_goal=0.05)
    leg024_traj = leg0_traj + leg2_traj + leg4_traj + body_traj
    # leg024_traj[:, 6] = 1
    start_time = time()
    for q in leg024_traj:
        hexy.viz.display(q)
        hexy.qc = q
    print(f"Time taken for this step = {time() - start_time}")
    leg0_traj = None
    leg2_traj = None
    leg4_traj = None
    body_traj = None
    sleep(3)
    step_size_xy_mult = 2
    leg0_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='S', LEG=0)
    leg2_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='S', LEG=2)
    leg4_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='S', LEG=4)
    body_traj = hexy.generate_body_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='S', FEET_GRP='135')
    leg024_traj = leg0_traj + leg2_traj + leg4_traj
    # leg024_traj[:, 6] = 1
    start_time = time()
    for q in leg024_traj:
        hexy.viz.display(q)
        hexy.qc = q
    print(f"Time taken for this step = {time() - start_time}")
