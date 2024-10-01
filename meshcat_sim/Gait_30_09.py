# we import useful libraries
from time import sleep, time
import numpy as np
import pinocchio as pin
import meshcat.geometry as g
import scipy
from pathlib import Path
import sys
import scipy.optimize 
from scipy.optimize import minimize

# we don't want to print every decimal!
np.set_printoptions(suppress=True, precision=4)


class hexapod:

    def __init__(self, init_viz=True) -> None:
        self.pin_model_dir = str(
            Path('URDF/URDF4Pin').absolute())
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
        self.state_flag = 'START'
        self.qc = self.robot.q0
        self.viz_flag = init_viz
        if self.viz_flag == True:
            self.init_viz()

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

    def inverse_geometery(self, q, leg_number, desired_pos):
        FOOT_ID = self.FOOT_IDS[leg_number]

        def foot_pos_err(joint_angles):
            q_updated = q.copy()
            index_leg = 7 + leg_number * 3
            q_updated[index_leg:index_leg+3] = joint_angles
            self.robot.forwardKinematics(q_updated)
            current_foot_pos = self.robot.framePlacement(q_updated, FOOT_ID).translation
            error = current_foot_pos - desired_pos
            return error.dot(error)

        index_leg = 7 + leg_number * 3
        joint_angles_initial = q[index_leg:index_leg+3]
        bounds = [(-np.pi/3, np.pi/3)] * 3
        res = scipy.optimize.minimize(foot_pos_err, joint_angles_initial, bounds=bounds)
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
    
    def compute_body_trajectory(self, v_body, T, N):
    
        Delta_t = T / N
        t_steps = np.linspace(0, T, N+1)
        body_positions = []
        for t in t_steps:
            p_body = self.qc[0:3].copy()
            p_body[0] += v_body * t  # Move along the x-axis
            body_positions.append(p_body)
        return np.array(body_positions), Delta_t

    def generate_joint_waypoints(self, step_size, body_positions, LEG, STEPS, t_init, t_goal):
        """
        Generate joint waypoints for a leg, considering the body movement.
        """
        # Foot start position relative to the world frame
        foot_start = self.robot.framePlacement(self.qc, self.FOOT_IDS[LEG]).translation

        # Body positions during the swing phase
        t_steps = np.linspace(t_init, t_goal, STEPS)
        foot_positions = []
        for idx, t in enumerate(t_steps):
            # Compute the foot position relative to the moving body
            body_p = body_positions[int((t / t_goal) * (body_positions.shape[0] - 1))]
            foot_p = foot_start + step_size * np.array([1, 0, 0])  # Move along x-axis
            foot_p[0] += v_body * t  # Adjust for body movement
            foot_positions.append(foot_p)
        # Use inverse kinematics to compute joint angles
        joint_waypoints = [self.inverse_geometry(self.qc, self.FOOT_IDS[LEG], fp) for fp in foot_positions]
        return joint_waypoints

    def update_body_position(self, v_body, Delta_t):
        """
        Update the body position based on the desired body velocity.
        """
        # Move the body forward along its x-axis
        self.qc[0] += v_body * Delta_t  # Assuming the body's x-axis is aligned with the world x-axis

    def inverse_geometry_stance_leg(self, q, leg_number, foot_position_world):
        FOOT_ID = self.FOOT_IDS[leg_number]

        def foot_pos_err_stance(joint_angles):
            q_updated = q.copy()
            index_leg = 7 + leg_number * 3
            q_updated[index_leg:index_leg+3] = joint_angles
            self.robot.forwardKinematics(q_updated)
            current_foot_pos = self.robot.framePlacement(q_updated, FOOT_ID).translation
            error = current_foot_pos - foot_position_world
            return error.dot(error)

        index_leg = 7 + leg_number * 3
        joint_angles_initial = q[index_leg:index_leg+3]
        bounds = [(-np.pi/3, np.pi/3)] * 3
        res = scipy.optimize.minimize(foot_pos_err_stance, joint_angles_initial, bounds=bounds)
        return res.x



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

    def visualize_trajectories(self, body_positions, foot_positions_list):
        """
        Visualize body and foot trajectories.
        """
        # Body trajectory
        self.viz.viewer['body_trajectory'].set_object(
            g.Line(g.PointsGeometry(body_positions.T), g.MeshBasicMaterial(color=0x00ff00))
        )
        # Foot trajectories
        colors = [0xff0000, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff, 0xffffff]
        for idx, foot_positions in enumerate(foot_positions_list):
            self.viz.viewer[f'foot_{idx}_trajectory'].set_object(
                g.Line(g.PointsGeometry(np.array(foot_positions).T), g.MeshBasicMaterial(color=colors[idx]))
            )

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

    def optimize_joint_trajectory(self, joint_waypoints):
        """
        Optimize joint trajectory to minimize changes in joint angles.
        """
        N = len(joint_waypoints)
        initial_guess = np.concatenate(joint_waypoints)
        
        def objective(x):
            x = x.reshape(N, -1)
            cost = 0
            for k in range(1, N):
                cost += np.linalg.norm(x[k] - x[k-1])**2
            return cost
    
        # Constraints and bounds can be added if necessary
        res = scipy.optimize.minimize(objective, initial_guess)
        optimized_joint_angles = res.x.reshape(N, -1)
        return optimized_joint_angles


    def optimize_body_velocity(self, desired_velocities, N=100, Delta_t=0.01):
        """
        Optimize body velocities over N time steps to follow desired velocities.
        """
        # Initial guess for variables: [v_body_0, omega_body_0, ..., v_body_N-1, omega_body_N-1]
        x0 = np.zeros(N * 6)
        
        # Bounds for velocities
        v_max = np.array([v_max_x, v_max_y, v_max_z])
        omega_max = np.array([omega_max_x, omega_max_y, omega_max_z])
        bounds = []
        for _ in range(N):
            bounds.extend([(-v_max[0], v_max[0]), (-v_max[1], v_max[1]), (-v_max[2], v_max[2])])
            bounds.extend([(-omega_max[0], omega_max[0]), (-omega_max[1], omega_max[1]), (-omega_max[2], omega_max[2])])

        # Objective function
        def objective(x):
            cost = 0
            for k in range(N):
                idx = k * 6
                v_body_k = x[idx:idx+3]
                omega_body_k = x[idx+3:idx+6]
                v_desired_k = desired_velocities['linear'][k]
                omega_desired_k = desired_velocities['angular'][k]
                cost += np.linalg.norm(v_body_k - v_desired_k)**2
                cost += alpha * np.linalg.norm(omega_body_k - omega_desired_k)**2
            return cost

        # Constraints
        constraints = []

        # Implement constraints for stability, kinematics, etc.

        # Solve the optimization problem
        res = minimize(objective, x0, bounds=bounds, constraints=constraints)
        
        # Extract optimized velocities
        optimized_velocities = {'linear': [], 'angular': []}
        x_opt = res.x
        for k in range(N):
            idx = k * 6
            optimized_velocities['linear'].append(x_opt[idx:idx+3])
            optimized_velocities['angular'].append(x_opt[idx+3:idx+6])
        
        return optimized_velocities

if __name__ == "__main__":
    hexy = hexapod(init_viz=True)
    sleep(3)
    v_body = 0.02  # Desired body velocity along x-axis
    T = 2.0       # Total time duration
    N = 100       # Number of time steps
    Delta_t = T / N

    # Initial foot positions in the world frame
    foot_positions_world = {}
    for leg in range(6):
        foot_positions_world[leg] = hexy.robot.framePlacement(hexy.qc, hexy.FOOT_IDS[leg]).translation.copy()

    # Gait sequence: stance legs and swing legs
    gait_sequence = [
        {'stance_legs': [0, 2, 4], 'swing_legs': [1, 3, 5]},
        {'stance_legs': [1, 3, 5], 'swing_legs': [0, 2, 4]},
    ]
    gait_phase_duration = N // 2  # Number of time steps per gait phase

    for phase in range(2):  # Two phases for the tripod gait
        stance_legs = gait_sequence[phase]['stance_legs']
        swing_legs = gait_sequence[phase]['swing_legs']

        # Generate swing leg trajectories
        swing_leg_trajectories = {}
        for leg in swing_legs:
            foot_start_pos = foot_positions_world[leg]
            step_length = 0.05
            foot_end_pos = foot_start_pos + np.array([step_length, 0, 0])
            t_steps = np.linspace(0, gait_phase_duration * Delta_t, gait_phase_duration)
            foot_positions = []
            for t in t_steps:
                s = t / (gait_phase_duration * Delta_t)
                foot_pos = (1 - s) * foot_start_pos + s * foot_end_pos
                foot_pos[2] += 0.02 * np.sin(np.pi * s)
                foot_positions.append(foot_pos)
            swing_leg_trajectories[leg] = foot_positions

        for k in range(gait_phase_duration):
            hexy.update_body_position(v_body, Delta_t)

            for leg in stance_legs:
                joint_angles = hexy.inverse_geometry_stance_leg(hexy.qc, leg, foot_positions_world[leg])
                index_leg = 7 + leg * 3
                hexy.qc[index_leg:index_leg+3] = joint_angles

            for leg in swing_legs:
                foot_pos = swing_leg_trajectories[leg][k]
                joint_angles = hexy.inverse_geometery(hexy.qc, leg, foot_pos)
                index_leg = 7 + leg * 3
                hexy.qc[index_leg:index_leg+3] = joint_angles

            if hexy.viz_flag:
                hexy.viz.display(hexy.qc)
            sleep(Delta_t)

        for leg in swing_legs:
            foot_positions_world[leg] = swing_leg_trajectories[leg][-1]


