# we import useful libraries
from time import sleep, time, strftime
import numpy as np
import pinocchio as pin
import meshcat.geometry as g
from pathlib import Path
import sys
from scipy.optimize import minimize
import logging

# we don't want to print every decimal!
np.set_printoptions(suppress=True, precision=4)


class hexapod:

    def __init__(self, init_viz=True, logging_level=logging.DEBUG) -> None:
        self.init_logger(logging_level)
        self.pin_model_dir = str(
            Path('./URDF/URDF4Pin').absolute())
        self.urdf_filename = (self.pin_model_dir + '/URDF4Pin.urdf')

        self.robot = pin.RobotWrapper.BuildFromURDF(
            self.urdf_filename, self.pin_model_dir, root_joint=pin.JointModelFreeFlyer())

        self.robot.forwardKinematics(pin.neutral(self.robot.model))
        self.robot.updateGeometryPlacements()
        self.direction_slope = 0.0
        self.HALF_STEP_SIZE_XY = 0.06/2
        self.STEP_SIZE_Z = 0.025
        self.FOOT_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "foot" in frame.name]
        self.SHOULDER_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "Revolute_joint_0" in frame.name]
        self.BASE_FRAME_ID = self.robot.model.getFrameId("robot_base")
        self.state_flag = 'START'
        self.qc = self.robot.q0
        self.viz_flag = init_viz
        if self.viz_flag == True:
            self.init_viz()
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
            sys.exit(0)

        # Load the robot in the viewer.
        self.viz.loadViewerModel()

        self.viz.displayFrames(visibility=True)
        self.viz.displayCollisions(visibility=False)
        self.viz.display(self.qc)

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
                f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")

    def find_frame_info_by_name(self, name: str):
        for frame in self.robot.model.frames:
            if name in frame.name:
                print(
                    f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")

    def inverse_geometery(self, q, FRAME_ID=9, desired_pos=np.zeros(3)):
        bounds = [(0., 0.)]*self.robot.nq
        bounds[0] = (self.qc[0], self.qc[0])
        bounds[1] = (self.qc[1], self.qc[1])
        bounds[2] = (self.qc[2], self.qc[2])
        bounds[3] = (self.qc[3], self.qc[3])
        bounds[4] = (self.qc[4], self.qc[4])
        bounds[5] = (self.qc[5], self.qc[5])
        bounds[6] = (self.qc[6], self.qc[6])
        for i in range(7, self.robot.nq):
            bounds[i] = ((-np.pi/3), (np.pi/3))
        res = minimize(
            self.foot_pos_err, q, args=(FRAME_ID, desired_pos), bounds=bounds)
        return res.x

    def north_vector(self):
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
        v = self.generate_direction_vector(DIR=DIR)
        p1 = self.robot.framePlacement(self.qc, FOOT_ID).translation[0:2]
        p2 = p1 + (self.HALF_STEP_SIZE_XY * step_size_xy_mult * v)
        self.x_t = lambda t: ((1 - t) * p1[0]) + (t * p2[0])
        self.y_t = lambda t:  ((1 - t) * p1[1]) + (t * p2[1])

        self.z_t = lambda t: (((-self.STEP_SIZE_Z)/0.25) *
                              (t ** 2)) + ((self.STEP_SIZE_Z / 0.25) * t)
        pass

    def generate_joint_waypoints(self, step_size_xy_mult, STEPS=5, DIR='N', FOOT_ID=10):
        self.init_foot_trajectory_functions(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, FOOT_ID=FOOT_ID)
        s = np.linspace(0, 1, STEPS)
        waypoints = [[round(self.x_t(t), 5), round(
            self.y_t(t), 5), round(self.z_t(t), 5)] for t in s]
        points = np.array([waypoints[0], waypoints[-1]]).T

        if self.viz_flag:
            self.viz.viewer[('Foot_ID_' + str(FOOT_ID) + '_trajectory')].set_object(
                g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xffff00)))

        return [self.inverse_geometery(self.qc, FOOT_ID, wp)
                for wp in waypoints]

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

    def generate_leg_joint_trajectory(self, step_size_xy_mult, DIR='N', LEG=0, STEPS=5, t_init=0, t_goal=0.1, dt=0.01):
        q_wps = self.generate_joint_waypoints(step_size_xy_mult,
                                              DIR=DIR, FOOT_ID=self.FOOT_IDS[LEG], STEPS=STEPS)
        q_traj = self.qc
        mask = np.concatenate((np.zeros(6), [1], np.zeros(
            LEG*3), [1, 1, 1], np.zeros((5-LEG)*3)))
        for i in range(0, q_wps.__len__()-1):
            t = t_init
            while t < t_goal:
                q_t = self.compute_trajectory(
                    q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
                q_traj = np.vstack((q_traj, np.multiply(q_t, mask)))
                t = (t + dt)
        return np.delete(q_traj, 0, axis=0)
        # return q_traj

    # In your hexapod class, add a method to get foot positions
    def get_foot_positions(self, q):
        return [self.robot.framePlacement(q, foot_id).translation for foot_id in self.FOOT_IDS]

    def feet_error(self, q_joints, desired_base_pose):
        # Record initial foot positions
        initial_foot_positions = self.get_foot_positions(self.qc)
        q_full = np.concatenate([desired_base_pose, q_joints])
        hexy.robot.forwardKinematics(q_full)
        error = 0
        for foot_id, desired_pos in zip(hexy.FOOT_IDS, initial_foot_positions):
            current_pos = hexy.robot.framePlacement(
                q_full, foot_id).translation
            error += np.linalg.norm(current_pos - desired_pos)**2
        return error

    def body_inverse_geometry(self, q, desired_base_pos):

        # Joint angle bounds (exclude base joints)
        bounds = [(-np.pi/3, np.pi/3)] * (self.robot.nq - 7)

        # Initial joint angles
        q_joints_init = q[7:].copy()

        res = minimize(
            self.feet_error,
            q_joints_init, args=(desired_base_pos),
            bounds=bounds,
            method='SLSQP', options={'disp': True}
        )

        return np.concatenate([desired_base_pos, res.x])

    def generate_body_path_waypoints(self, step_size_xy_mult=1, STEPS=5, DIR='N'):
        self.init_foot_trajectory_functions(
            step_size_xy_mult=step_size_xy_mult, DIR=DIR, FOOT_ID=self.BASE_FRAME_ID)
        s = np.linspace(0, 1, STEPS)
        waypoints = [np.concatenate(([round(self.x_t(t), 5), round(
            self.y_t(t), 5)], self.qc[2:7].copy())) for t in s]

        points = np.array(waypoints)[:, 0:3].T
        if self.viz_flag:
            self.viz.viewer[('Base_trajectory')].set_object(
                g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xffff00)))
        return [self.body_inverse_geometry(self.qc, wp)
                for wp in waypoints]

    def generate_body_joint_trajectory(self, step_size_xy_mult, DIR='N', STEPS=5, t_init=0, t_goal=0.1, dt=0.01):
        q_wps = self.generate_body_path_waypoints(step_size_xy_mult,
                                                  DIR=DIR, STEPS=STEPS)
        q_traj = self.qc
        for i in range(0, q_wps.__len__()-1):
            t = t_init
            while t < t_goal:
                q_t = self.compute_trajectory(
                    q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
                q_traj = np.vstack((q_traj, q_t))
                t = (t + dt)
        return np.delete(q_traj, 0, axis=0)
        # return q_traj


# if __name__ == "__main__":
#     hexy = hexapod(init_viz=True)
#     sleep(3)
#     step_size_xy_mult = 1
#     leg0_traj = hexy.generate_leg_joint_trajectory(
#         step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=0)
#     leg2_traj = hexy.generate_leg_joint_trajectory(
#         step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=2)
#     leg4_traj = hexy.generate_leg_joint_trajectory(
#         step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=4)
#     leg024_traj = leg0_traj + leg2_traj + leg4_traj
#     leg024_traj[:, 6] = 1
#     for q in leg024_traj:
#         hexy.viz.display(q)
#         hexy.qc = q
#     leg0_traj = None
#     leg2_traj = None
#     leg4_traj = None
#     sleep(3)
#     step_size_xy_mult = 2
#     leg0_traj = hexy.generate_leg_joint_trajectory(
#         step_size_xy_mult=step_size_xy_mult, DIR='S', LEG=0)
#     leg2_traj = hexy.generate_leg_joint_trajectory(
#         step_size_xy_mult=step_size_xy_mult, DIR='S', LEG=2)
#     leg4_traj = hexy.generate_leg_joint_trajectory(
#         step_size_xy_mult=step_size_xy_mult, DIR='S', LEG=4)
#     leg024_traj = leg0_traj + leg2_traj + leg4_traj
#     leg024_traj[:, 6] = 1
#     for q in leg024_traj:
#         hexy.viz.display(q)
#         hexy.qc = q


# At the end of your main function or script
if __name__ == "__main__":
    hexy = hexapod(init_viz=True, logging_level=logging.DEBUG)
    step_size_xy_mult = 1
    leg0_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=0)
    leg2_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=2)
    leg4_traj = hexy.generate_leg_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=4)
    body_traj = hexy.generate_body_joint_trajectory(
        step_size_xy_mult=step_size_xy_mult, DIR='N')
    q = np.hstack((body_traj[:, 0:7], leg0_traj[:, 7:10], body_traj[:, 10:13],
                   leg2_traj[:, 13:16], body_traj[:, 16:19], leg4_traj[:, 19:22], body_traj[:, 22:25]))
    sleep(3)
    hexy.viz.play(q)
    hexy.qc = q[-1]
    hexy.logger.info(
        'First step taken, below code implements the second step')
    step_size_xy_mult = 2
    while (1):
        leg1_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=1)
        leg3_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=3)
        leg5_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=5)
        body_traj = hexy.generate_body_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N')
        q = np.hstack((
            body_traj[:, 0:7], body_traj[:, 7:10], leg1_traj[:, 10:13],
            body_traj[:, 13:16], leg3_traj[:,
                                           16:19], body_traj[:, 19:22], leg5_traj[:, 22:25]
        ))
        hexy.viz.play(q)
        hexy.qc = q[-1]
        leg0_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=0)
        leg2_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=2)
        leg4_traj = hexy.generate_leg_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=4)
        body_traj = hexy.generate_body_joint_trajectory(
            step_size_xy_mult=step_size_xy_mult, DIR='N')
        q = np.hstack((body_traj[:, 0:7], leg0_traj[:, 7:10], body_traj[:, 10:13],
                       leg2_traj[:, 13:16], body_traj[:, 16:19], leg4_traj[:, 19:22], body_traj[:, 22:25]))
        hexy.viz.play(q)
        hexy.qc = q[-1]
