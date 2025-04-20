import meshcat.geometry as g
import pinocchio as pin
from pathlib import Path
from pinocchio import JointModelFreeFlyer 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.set_printoptions(suppress=True, precision=6)

class Hexapod:
    # This class is supposed to provide the shits required to finish this stupid ass project

    def __init__(self, home_visualizer=True,debuging_mode=True) -> None:
        
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
        #    to which we feed a vector of 43 elements aka state vector: 
        #    [0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.
        #     0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0.]
        #    Here the first 7 suggest the home config of the base or root link (X_Y_Z_w_x_y_z(Quaternionangles)) and
        #    the remaining 36 elements corresponds to the 18 joints exculding any fixed joint.
        # 4. Next we move on to initialising the step size for the leg as 0.06m along XY plane and 0.025m as the vertical height (Z). 
        #    The trajectory shaped by each leg is that of a parabolla (y=−27.78(x^2)+0.025)
        # 5. Each of our robot elements have been assigned frames as listed below. From which we have to extract the shoulder, foot and base IDs.
        # 6. We create a varialble that store the currrent position of the joint and initialise it with home joint configurations ie; the 
        #    state vector.
        # 7. Last we initialize our visualiser Meshcat
        
        self.Directory_path=str(Path('./URDF/Physics_Model_Edited').absolute())
        self.urdf_filename=(self.Directory_path +
                              '/Physics Model URDF (Edtd.).urdf')
        
        try:
            self.robot = pin.RobotWrapper.BuildFromURDF(
                self.urdf_filename,
                self.Directory_path,
                root_joint=JointModelFreeFlyer()
            )
            print("URDF successfully loaded.")
        except Exception as e:
            print(f"Failed to build RobotWrapper XXXXX: {e}")
            raise
        
        self.robot.forwardKinematics(pin.neutral(self.robot.model))
              
        self.HALF_STEPSIZE_XY=0.03
        self.Vert_STEPSIZE_Z=0.03
        
        self.FOOT_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "foot" in frame.name]
        self.SHOULDER_IDS = [self.robot.model.getFrameId(
            frame.name) for frame in self.robot.model.frames if "Revolute_joint_0" in frame.name]
        self.BASE_FRAME_ID = self.robot.model.getFrameId("robot_base")
        self.qc = self.robot.q0   

        # State Vector (1X49)
        self.statevector_initial = self.create_state_vector(self.robot.q0)
        self.statevector_current = self.create_state_vector(self.qc)
        
        if debuging_mode:

            print(self.statevector_initial)      
            print(self.statevector_current)

        if debuging_mode:
          '''-------------------------------------------------------------'''
          '''___________________Frame IDs of the Joints___________________'''
          # Use only For Debugging
          for frame in self.robot.model.frames:
                  # Get the frame ID
                  frame_id = self.robot.model.getFrameId(frame.name)
                  
                  # Get the placement of the frame (rotation and translation)
                  placement = self.robot.framePlacement(self.qc, frame_id)

                  # Extract the rotation matrix (3x3) and translation vector (3x1)
                  R = placement.rotation  # 3x3 rotation matrix
                  p = placement.translation  # 3x1 translation vector

                  # Create a 4x4 transformation matrix
                  T = np.eye(4)
                  T[:3, :3] = R  # Set the rotation part
                  T[:3, 3] = p   # Set the translation part

                  # Print the frame information and the transformation matrix
                  print(f"Frame Name: {frame.name}")
                  print(f"Frame Type: {frame.type}")
                  print(f"Frame ID: {frame_id}")
                  print("Transformation Matrix (4x4):")
                  print(T)
                  print()  # For better readability

          '''___________________Frame ID List___________________'''
            # Frame ID:  0, Name: universe (Represents the origin or world frame)
            # Frame ID:  1, Name: root_joint (The free-flyer joint that allows the base of the robot to move)
            # Frame ID:  2, Name: robot_base
            # Frame ID:  3, Name: Revolute_joint_01
            # Frame ID:  4, Name: coxa_1
            # Frame ID:  5, Name: Revolute_joint_11
            # Frame ID:  6, Name: femur_1
            # Frame ID:  7, Name: Revolute_joint_21
            # Frame ID:  8, Name: tibia_1
            # Frame ID:  9, Name: Spherical_joint_31
            # Frame ID: 10, Name: foot_1
            # Frame ID: 11, Name: Revolute_joint_02
            # Frame ID: 12, Name: coxa_2
            # Frame ID: 13, Name: Revolute_joint_12
            # Frame ID: 14, Name: femur_2
            # Frame ID: 15, Name: Revolute_joint_22
            # Frame ID: 16, Name: tibia_2
            # Frame ID: 17, Name: Spherical_joint_32
            # Frame ID: 18, Name: foot_2
            # Frame ID: 19, Name: Revolute_joint_03
            # Frame ID: 20, Name: coxa_3
            # Frame ID: 21, Name: Revolute_joint_13
            # Frame ID: 22, Name: femur_3
            # Frame ID: 23, Name: Revolute_joint_23
            # Frame ID: 24, Name: tibia_3
            # Frame ID: 25, Name: Spherical_joint_33
            # Frame ID: 26, Name: foot_3
            # Frame ID: 27, Name: Revolute_joint_04
            # Frame ID: 28, Name: coxa_4
            # Frame ID: 29, Name: Revolute_joint_14
            # Frame ID: 30, Name: femur_4
            # Frame ID: 31, Name: Revolute_joint_24
            # Frame ID: 32, Name: tibia_4
            # Frame ID: 33, Name: Spherical_joint_34
            # Frame ID: 34, Name: foot_4
            # Frame ID: 35, Name: Revolute_joint_05
            # Frame ID: 36, Name: coxa_5
            # Frame ID: 37, Name: Revolute_joint_15
            # Frame ID: 38, Name: femur_5
            # Frame ID: 39, Name: Revolute_joint_25
            # Frame ID: 40, Name: tibia_5
            # Frame ID: 41, Name: Spherical_joint_35
            # Frame ID: 42, Name: foot_5
            # Frame ID: 43, Name: Revolute_joint_06
            # Frame ID: 44, Name: coxa_6
            # Frame ID: 45, Name: Revolute_joint_16
            # Frame ID: 46, Name: femur_6
            # Frame ID: 47, Name: Revolute_joint_26
            # Frame ID: 48, Name: tibia_6
            # Frame ID: 49, Name: Spherical_joint_36
            # Frame ID: 50, Name: foot_6
          '''-------------------------------------------------------------'''      

        self.viz_flag = home_visualizer
        if self.viz_flag == True:
            self.home_visualizer()
        if debuging_mode:
          print("DEBUG POINT")
        
        

        
    def create_state_vector(self, q, ):
        """
        Function creates the state vectors required for optimization
        """
        self.robot.forwardKinematics(q)
        statevector = np.zeros((1, 49))
        statevector[0, 1:7] = q[1:7]  # Assign the base orientation (quaternion)
        i = 0
        for foot_id in self.FOOT_IDS:
            placement = self.robot.framePlacement(q, foot_id)
            position = placement.translation  # Extract XYZ position
            quaternion = pin.Quaternion(placement.rotation)  # Convert rotation to quaternion
            
            # Assign position and quaternion to the state vector
            statevector[0, i + 7:i + 10] = position
            statevector[0, i + 10:i + 14] = quaternion.coeffs()
            i += 7
        
        return statevector      
  
    def home_visualizer(self):
      
        self.viz = pin.visualize.MeshcatVisualizer(
            self.robot.model, 
            self.robot.collision_model, 
            self.robot.visual_model
        )
        self.viz.initViewer(open=True) 
        self.viz.loadViewerModel()
        self.viz.displayFrames(visibility=True)
        self.viz.displayCollisions(visibility=False)
        self.viz.display(self.qc) 
   
        return 
    
    def quaternion_multiply(self, Q1, Q2):
        """
        Multiply two quaternions using the formula:
        Q1 * Q2 = {η1 η2 - ε1^T ε2, η1 ε2 + η2 ε1 + ε1 × ε2}

        Args:
            Q1 (tuple): First quaternion (η1, ε1) where ε1 is a NumPy array.
            Q2 (tuple): Second quaternion (η2, ε2) where ε2 is a NumPy array.

        Returns:
            tuple: Resulting quaternion (η, ε) as (scalar, vector)
        """

        # Extract scalar and vector parts
        eta1, eps1 = Q1
        eta2, eps2 = Q2

        # Compute quaternion multiplication
        eta = eta1 * eta2 - np.dot(eps1, eps2)  # Scalar part
        eps = eta1 * eps2 + eta2 * eps1 + np.cross(eps1, eps2)  # Vector part

        return (eta, eps)
    
    def cost_twist(self,q_joints,desired_base_pose,q):
        
        self.robot.forwardKinematics(q)
        current_placement = self.robot.framePlacement(q, self.BASE_FRAME_ID).rotation
        raw_quaternion = pin.Quaternion(current_placement)
        current_quaternion = np.array(raw_quaternion.coeffs())
        print(f'current_quaternion: {current_quaternion}')
        Q1 = (current_quaternion[3],current_quaternion[:3])
        
        Quat_fr_rotation = np.array([0, 0, 0.5*np.sqrt(2-2*np.cos(0.261799)), 0.5*(np.sqrt(2+2*np.cos(0.261799)))])
        print(f'Quat_fr_rotation: {Quat_fr_rotation}')
        Q2 = (Quat_fr_rotation[3],Quat_fr_rotation[:3])
        
        desired_quaternion = self.quaternion_multiply(Q1, Q2)
        print(f'Q_d = {desired_quaternion}')
        
        
        # To compute error we use the formula: ΔQ = Qd*Qc^-1
        Q_inv = (current_quaternion[3],-1*current_quaternion[:3])
        print(f'current_quaternion (inv): {Q_inv}')
        error = self.quaternion_multiply(Q1 = desired_quaternion, Q2 = Q_inv)
        print(f'Quaternion error = {error}')
    
    
    def print_all_frame_info(self) -> None:
       """
       Print information about all frames in the robot model.
       """
       for frame in self.robot.model.frames:
           # Print the frame name, type, ID, and position
           print(
               f"Frame Name: {frame.name}, Frame Type: {frame.type}, Frame ID: {self.robot.model.getFrameId(frame.name)}, XYZ = {self.robot.framePlacement(self.qc, self.robot.model.getFrameId(frame.name))}")
    
    
    
    def foot_pos_err(self, q, FRAME_ID, desired_pos=np.zeros(3)):
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
        # Set bounds for optimization (fix the base joint positions)
        bounds = [(0., 0.)]*self.robot.nq
        bounds[0] = (self.qc[0], self.qc[0])  # x position of base
        bounds[1] = (self.qc[1], self.qc[1])  # y position of base
        bounds[2] = (self.qc[2], self.qc[2])  # z position of base
        bounds[3] = (self.qc[3], self.qc[3])  # Quaternion component
        bounds[4] = (self.qc[4], self.qc[4])  # Quaternion component
        bounds[5] = (self.qc[5], self.qc[5])  # Quaternion component
        bounds[6] = (self.qc[6], self.qc[6])  # Quaternion component
        for i in range(7, self.robot.nq):
            # Joint limits for other joints
            bounds[i] = ((-np.pi/3), (np.pi/3))
        # Perform minimization to find joint configuration minimizing foot position error
        res = minimize(
            self.foot_pos_err, q, args=(FRAME_ID, desired_pos), bounds=bounds)
        # Return the optimized joint configuration
        return res.x
  
    def create_circle_and_next_point(self, FOOT_IDS=10, debug=False):
        # Extract circle center and radius from three foot positions
        p1 = self.robot.framePlacement(self.qc, self.FOOT_IDS[0]).translation[:2]
        p2 = self.robot.framePlacement(self.qc, self.FOOT_IDS[2]).translation[:2]
        p3 = self.robot.framePlacement(self.qc, self.FOOT_IDS[4]).translation[:2]

        def find_circle_equation(p1, p2, p3):
            # Extract coordinates
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3

            # Set up the system of equations
            A = np.array([
                [2 * (x2 - x1), 2 * (y2 - y1)],
                [2 * (x3 - x1), 2 * (y3 - y1)]
            ])
            B = np.array([
                x2**2 - x1**2 + y2**2 - y1**2,
                x3**2 - x1**2 + y3**2 - y1**2
            ])

            # Solve for h and k (center of the circle)
            h, k = np.linalg.solve(A, B)

            # Calculate the radius
            r = np.sqrt((x1 - h)**2 + (y1 - k)**2)

            return np.round(h, 6), np.round(k, 6), np.round(r, 10)

        # Create circle
        h, k, r = find_circle_equation(p1, p2, p3)
        circumference = 2 * np.pi * r
        self.xp1 = self.robot.framePlacement(self.qc, FOOT_IDS).translation[:3] 
        print(f'xp1 = {self.xp1[0]}')

        def find_second_point_on_circle(circumference, R, clockwise=True):
            """
            Finds the second point on a circle of radius R centered at (h, k),
            at a fixed arc distance from P1 in a clockwise direction.
            """
            x1 = self.xp1[0]
            y1 = self.xp1[1]
            theta1 = np.arctan2(y1 - k, x1 - h)  # Initial angle from center to p1
            arc_length = circumference / 20
            delta_theta = -arc_length / R if clockwise else arc_length / R
            theta2 = theta1 + delta_theta
            x2 = h + R * np.cos(theta2)
            y2 = k + R * np.sin(theta2)
            p2 = (x2, y2)
            direction_vector = np.array([x2 - x1, y2 - y1])
            return p2, direction_vector

        # Compute the next point and direction vector
        self.next_point, self.direction_vector = find_second_point_on_circle(circumference, r, clockwise=True)

        if debug:
    
            theta_vals = np.linspace(0, 2 * np.pi, 500)
            circle_x = r * np.cos(theta_vals) + h
            circle_y = r * np.sin(theta_vals) + k

            p1 = self.robot.framePlacement(self.qc, FOOT_IDS).translation[:2]  # Ensure updated p1
            print(f"Updated p1 for plotting: {p1}")
            print(f"Updated direction_vector for plotting: {self.direction_vector}")

            plt.figure(figsize=(8, 8))
            plt.plot(circle_x, circle_y, label="Circle", color="blue")
            plt.scatter(*p1, color="red", label="First Point (P1)")
            plt.scatter(*self.next_point, color="green", label="Second Point (P2)")

            plt.quiver([p1[0]], [p1[1]], [self.direction_vector[0]], [self.direction_vector[1]], 
                       angles='xy', scale_units='xy', scale=1, color="purple", label="Direction Vector")

            plt.axhline(0, color="gray", linewidth=0.5)
            plt.axvline(0, color="gray", linewidth=0.5)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim(h - r - 0.1, h + r + 0.1)
            plt.ylim(k - r - 0.1, k + r + 0.1)
            plt.title("Circle and Next Point with Direction Vector")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.legend()
            plt.grid()
            plt.show()


        # Normalize the rotated vector
            norm_direction_vector = np.linalg.norm(self.direction_vector)
            if norm_direction_vector == 0:
                raise ValueError("The direction vector v must be non-zero.")
            self.V = self.direction_vector / norm_direction_vector

            if self.viz_flag:
                # Assuming FOOT_IDS is a list and FOOT_IDS=10 refers to the 10th index
                start_point = self.robot.framePlacement(self.qc, FOOT_IDS).translation
                end_point = start_point + np.append(self.V, 0) * 0.1   # Adjust scaling as needed

                print("Direction Vector Start Point:", start_point)
                print("Direction Vector End Point:", end_point)

                self.viz.viewer['direction_vector'].set_object(
                    g.Line(g.PointsGeometry(np.array([start_point, end_point]).T), 
                           g.MeshBasicMaterial(color=0xffff00))
                )

    def trajectory_discretized (self,FOOT_IDS,xytransalation = 0.0908,debug = False):
        
        self.create_circle_and_next_point(FOOT_IDS=FOOT_IDS,debug=True)
        print(f"FOOT_IDS (traj_discre): {FOOT_IDS}")
        swivel_start_point = self.robot.framePlacement(self.qc, FOOT_IDS).translation
        swivel_to_point = np.append(self.robot.framePlacement(self.qc, FOOT_IDS).translation[:2] + xytransalation*self.V, self.robot.framePlacement(self.qc, FOOT_IDS).translation[2])
        
        y_clearance = 0.03           # Vertical clearance for y apex
        z_clearance = 0.03           # Lateral deviation for z apex
        
        # Apex points (midway)
        x_apex = (swivel_start_point[0] + swivel_to_point [0]) / 2
        y_apex = max(swivel_start_point[1], swivel_to_point [1]) + (y_clearance * np.sign(max(swivel_start_point[1], swivel_to_point [1])))
        z_apex = max(swivel_start_point[2], swivel_to_point [2]) + (z_clearance * np.sign(max(swivel_start_point[2], swivel_to_point [2])))
        
        # Solve for parabolic coefficients for y(t) = a_y * t^2 + b_y * t + c_y
        A_y = np.array([
            [0**2, 0, 1],  # y(0) = y0
            [0.5**2, 0.5, 1],  # y(0.5) = y_apex
            [1**2, 1, 1]   # y(1) = y1
        ])
        b_y = np.array([swivel_start_point[1], y_apex, swivel_to_point [1]])
        a_y, b_y, c_y = np.linalg.solve(A_y, b_y)

        # Solve for parabolic coefficients for z(t) = a_z * t^2 + b_z * t + c_z
        A_z = np.array([
            [0**2, 0, 1],  # z(0) = z0
            [0.5**2, 0.5, 1],  # z(0.5) = z_apex
            [1**2, 1, 1]   # z(1) = z1
        ])
        b_z = np.array([swivel_start_point[2], z_apex, swivel_to_point [2]])
        a_z, b_z, c_z = np.linalg.solve(A_z, b_z)
        
        
        self.x_t = lambda t: (1 - t) * swivel_start_point[0] + t * swivel_to_point[0]
        self.y_t = lambda t: a_y * t**2 + b_y * t + c_y
        self.z_t = lambda t: a_z * t**2 + b_z * t + c_z
        
        if debug:
            print(f'swivel_start_point: {swivel_start_point}')
            print(f'swivel_to_point   : {swivel_to_point }')
        
        
    def generate_joint_waypoints(self, step_size_xy_mult=1, STEPS=5,FOOT_IDS=10):
        
        
        self.trajectory_discretized (FOOT_IDS)
        timesteps =np.linspace(0,1,STEPS)
        waypoints = [[round(self.x_t(t), 5), round(self.y_t(t), 5), round(self.z_t(t),5)] for t in timesteps]
        print(f'waypoints: {waypoints}')
        
        # Convert waypoints to array for visualization
        points = np.array([waypoints[0], waypoints[-1]]).T
        
        # Visualize the foot trajectory if visualization is enabled
        if self.viz_flag:
            self.viz.viewer[('Foot_ID_' + str(FOOT_IDS) + '_trajectory')].set_object(
                g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xFF0000)))
        
        return [self.inverse_geometery(self.qc, FOOT_IDS, wp)
                for wp in waypoints]
        
    
    def compute_trajectory_p(self, position_init, position_goal, t_init, t_goal, t):
        
        T = t_goal - t_init
        if T <= 0:
            raise ValueError("t_goal must be greater than t_init.")

        delta_t = t - t_init
        theta_diff = position_goal - position_init

        # Clamp delta_t to the range [0, T] to handle times outside the trajectory duration
        delta_t_clamped = np.clip(delta_t, 0.0, T)

        # Compute the fraction of time elapsed (s)
        fraction = delta_t_clamped / T

        # Compute desired position using linear interpolation
        self.desired_position = position_init + fraction * theta_diff

        # Compute desired velocity (constant)
        self.desired_velocity = theta_diff / T

        # Compute desired acceleration (zero)
        self.desired_acceleration = np.zeros_like(position_init)

        return self.desired_position, self.desired_velocity, self.desired_acceleration
        
def generate_leg_joint_trajectory(self, step_size_xy_mult, DIR='N', LEG=0, STEPS=5, t_init=0, t_goal=0.1, dt=0.01):
        
        # Generate joint waypoints for the specified leg
        q_wps = self.generate_joint_waypoints(step_size_xy_mult,DIR=DIR, FOOT_ID=self.FOOT_IDS[LEG], STEPS=STEPS)
        # self.plot_trajctory(title=f'Trajectory for Leg {LEG} after IK Before Interpolation', state=np.array(
        #     [self.forward_kinematics(q) for q in q_wps]))
        q_traj = self.qc
        
        # Create a mask to apply joint configurations to the specific leg
        mask = np.concatenate((np.zeros(6), [1], np.zeros(LEG*3), [1, 1, 1], np.zeros((5-LEG)*3)))
        for i in range(0, len(q_wps)-1):
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
          

def main():
    home_visualizer = True
    debuging_mode = True
    hex = Hexapod(home_visualizer, debuging_mode)
    
    desired_base_pose = np.array([0, 0, 0, 0, 0, 1, 0])  # Example pose
    print(f'desired_base_pose = {desired_base_pose}')
    q_joints = np.zeros(hex.robot.nq-7)  # Assuming zero joint angles
    print(f'q_joints =  {q_joints}')
    q = np.concatenate([desired_base_pose, q_joints])  # Full state vector
    print(f'q =  {q}')
    
    # Call the function
    quaternion = hex.cost_twist(q_joints, desired_base_pose, q)
    hex.print_all_frame_info()
       
    point1=hex.robot.framePlacement(hex.qc, hex.FOOT_IDS[0]).translation[:3]
    point2=hex.robot.framePlacement(hex.qc, hex.FOOT_IDS[2]).translation[:3]
    point3=hex.robot.framePlacement(hex.qc, hex.FOOT_IDS[4]).translation[:3]
    print(f"Foot id 0 = {point1}")
    print(f"Foot id 2 = {point2}")
    print(f"Foot id 4 = {point3}")
  
    a = hex.generate_joint_waypoints(step_size_xy_mult=1, FOOT_IDS=10, STEPS=5)
    b = hex.generate_joint_waypoints(step_size_xy_mult=1, FOOT_IDS=26, STEPS=5)
    c = hex.generate_joint_waypoints(step_size_xy_mult=1, FOOT_IDS=42, STEPS=5)
    
    print(f"Inverse Configs of foot 10: {a}")
    print(f"Inverse Configs of foot 26: {b}")
    print(f"Inverse Configs of foot 42: {c}")
 

if __name__ == "__main__":
    main()

   


