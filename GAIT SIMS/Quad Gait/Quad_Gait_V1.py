import meshcat.geometry as g
import pinocchio as pin
from pathlib import Path
from pinocchio import JointModelFreeFlyer 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from time import strftime

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
              
        self.HALF_STEPSIZE_XY=0.005
        self.Vert_STEPSIZE_Z=0.005
        
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
        
    def cost(self,q,FRAME_ID, desired_pos=np.zeros(3)):
       
       self.robot.forwardKinematics(q)
       current_pos = self.robot.framePlacement(q,FRAME_ID).translation
       error = desired_pos - current_pos
       
       return np.dot(error,error)
        
    def inverse_geometry_stable(self,q,FRAME_ID,desired_pos=np.zeros(3) ):
        
        '''This function is created to minimize the cost function and provide the inverse joint configurations in the process'''
        bounds = [(0,0)]*self.robot.nq
        bounds[0]   = (self.qc[0],self.qc[0])
        bounds[1]   = (self.qc[1],self.qc[1]) 
        bounds[2]   = (self.qc[2],self.qc[2]) 
        bounds[3]   = (self.qc[3],self.qc[3]) 
        bounds[4]   = (self.qc[4],self.qc[4]) 
        bounds[5]   = (self.qc[5],self.qc[5]) 
        bounds[6]   = (self.qc[6],self.qc[6]) 
        
        bounds[7]   = (-85 * np.pi / 180, 85 * np.pi / 180)
        bounds[8]   = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[9]   = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[10]  = (-85 * np.pi / 180, 85 * np.pi / 180)
        bounds[11]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[12]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[13]  = (self.qc[13],self.qc[13]) 
        bounds[14]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[15]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[16]  = (-85 * np.pi / 180, 85 * np.pi / 180)
        bounds[17]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[18]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[19]  = (-85 * np.pi / 180, 85 * np.pi / 180)
        bounds[20]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[21]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[22]  = (self.qc[22],self.qc[22]) 
        bounds[23]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[24]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        
        res = minimize(
            self.cost, q, args=(FRAME_ID, desired_pos), bounds=bounds, tol=1e-12, method='L-BFGS-B')
        
        return res.x
    
    def inverse_geometry_legmotion(self,q,FRAME_ID,desired_pos=np.zeros(3) ):
        
        '''This function is created to minimize the cost function and provide the inverse joint configurations in the process'''
        bounds = [(0,0)]*self.robot.nq
        bounds[0]   = (self.qc[0],self.qc[0])
        bounds[1]   = (self.qc[1],self.qc[1]) 
        bounds[2]   = (self.qc[2],self.qc[2]) 
        bounds[3]   = (self.qc[3],self.qc[3]) 
        bounds[4]   = (self.qc[4],self.qc[4]) 
        bounds[5]   = (self.qc[5],self.qc[5]) 
        bounds[6]   = (self.qc[6],self.qc[6]) 
        
        bounds[7]   = (-85 * np.pi / 180, 85 * np.pi / 180)
        bounds[8]   = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[9]   = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[10]  = (-85 * np.pi / 180, 85 * np.pi / 180)
        bounds[11]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[12]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[13]  = (self.qc[13],self.qc[13]) 
        bounds[14]  = (self.qc[14],self.qc[14])
        bounds[15]  = (self.qc[15],self.qc[15])
        bounds[16]  = (-85 * np.pi / 180, 85 * np.pi / 180)
        bounds[17]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[18]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[19]  = (-85 * np.pi / 180, 85 * np.pi / 180)
        bounds[20]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[21]  = (-45 * np.pi / 180, 45 * np.pi / 180)
        bounds[22]  = (self.qc[22],self.qc[22]) 
        bounds[23]  = (self.qc[23],self.qc[23])
        bounds[24]  = (self.qc[24],self.qc[24])
        
        res = minimize(
            self.cost, q, args=(FRAME_ID, desired_pos), bounds=bounds, tol=1e-12, method='L-BFGS-B')
        
        return res.x
    
    def quad_mode_lift_vect(self, Frame_ID, d=0.09575, target_z=0.10945):
       
         # Step 1: Define the origin explicitly
         origin = self.robot.framePlacement(self.qc, self.BASE_FRAME_ID).translation
         
         print(f"origin = {origin}")
         # Step 2: Get the current frame position (p1)
         p1 = self.robot.framePlacement(self.qc, Frame_ID).translation

         # Step 3: Direction vector from origin to p1 (unit vector)
         dir_to_p1 = -(p1 - origin) / np.linalg.norm(p1 - origin)

         # Step 4: Compute p2 at distance 'd' from p1 in dir_to_p1 direction, with fixed z
         displacement = dir_to_p1 * d
         
         p2 = p1 - displacement
         p2[2] = target_z  # override z-axis height
         lift_pose =p2
         print(p2)
         
         # Step 5: Compute direction vector from p1 to p2 (unit vector)
         v = p2 - p1
         dir_v = v / np.linalg.norm(v)

         # Step 6: Visualize direction vector in MeshCat if enabled
         if self.viz_flag:
             self.viz.viewer['direction_vector'].set_object(
                 g.Line(
                     g.PointsGeometry(np.array([p1, p1 + dir_v]).T),
                     g.MeshBasicMaterial(color=0xff0000)
                 )
             )

         return dir_v,lift_pose

     
    def quad_mode_initial_vect(self, Frame_ID):
        
        def rotate_vector(v, angle_rad):
            """Rotate vector v by angle in radians (2D only)"""
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad),   0],
                [np.sin(angle_rad),  np.cos(angle_rad),   0],
                [                0,                  0,   1]
            ])
            return rotation_matrix @ v
        
        center = self.robot.framePlacement(self.qc, (Frame_ID-6)).translation[:2]
        p1     = self.robot.framePlacement(self.qc, (Frame_ID)).translation
        center = np.append(center,p1[2])
        print(f"Frame_ID={Frame_ID}")
        if Frame_ID == 10 or Frame_ID == 34:
            angle = 30
        elif Frame_ID == 18 or Frame_ID == 42:
            angle = -30

            
        print(angle)
        angle_rad = np.deg2rad(angle)
        v1=p1-center
        
        dir_v = rotate_vector(v1, angle_rad)
        
        initial_pos =center + dir_v
                
        return dir_v,  initial_pos
    
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
            'I': self.quad_mode_initial_vect,
            'L': self.quad_mode_lift_vect,
            
        }
        return slope_switch_dict.get(DIR, self.default_vector)()

    def trajectory_discretized (self,Frame_ID, q_start=None,debug = False):
        
        if q_start is None:
            q_start = self.qc
        
        self.quad_mode_lift_vect(Frame_ID)
        # print(f"FOOT_IDS (traj_discre): {Frame_ID}")
        
        swing_start_point = self.robot.framePlacement(q_start, Frame_ID).translation
        _,swing_to_point = self.quad_mode_lift_vect(Frame_ID)
        
        self.x_t = lambda t: (1 - t) * swing_start_point[0] + t * swing_to_point[0]
        self.y_t = lambda t: (1 - t) * swing_start_point[1] + t * swing_to_point[1]
        self.z_t = lambda t: (1 - t) * swing_start_point[2] + t * swing_to_point[2]
        if debug:
            print(f'swivel_start_point: {swing_start_point}')
            print(f'swivel_to_point   : {swing_to_point }')
            
    def init_foot_trajectory_functions_stable(self, Frame_ID=10,q_start=None,debug = False):
        
        
        if q_start is None:
            q_start = self.qc
        
        self.quad_mode_initial_vect(Frame_ID)
        # print(f"FOOT_IDS (traj_discre): {Frame_ID}")
        
        swing_start_point = self.robot.framePlacement(q_start, Frame_ID).translation
        _,swing_to_point = self.quad_mode_initial_vect(Frame_ID)

        y_clearance = 0.03           # Vertical clearance for y apex
        z_clearance = 0.03           # Lateral deviation for z apex

        # Apex points (midway)
        y_apex = max(swing_start_point[1], swing_to_point[1]) + (y_clearance * np.sign(max(swing_start_point[1], swing_to_point[1])))
        z_apex = max(swing_start_point[2], swing_to_point[2]) + (z_clearance * np.sign(max(swing_start_point[2], swing_to_point[2])))

        # Solve for parabolic coefficients for y(t) = a_y * t^2 + b_y * t + c_y
        A_y = np.array([
            [0**2, 0, 1],  # y(0) = y0
            [0.5**2, 0.5, 1],  # y(0.5) = y_apex
            [1**2, 1, 1]   # y(1) = y1
        ])
        b_y = np.array([swing_start_point[1], y_apex, swing_to_point[1]])
        a_y, b_y, c_y = np.linalg.solve(A_y, b_y)

        # Solve for parabolic coefficients for z(t) = a_z * t^2 + b_z * t + c_z
        A_z = np.array([
            [0**2, 0, 1],  # z(0) = z0
            [0.5**2, 0.5, 1],  # z(0.5) = z_apex
            [1**2, 1, 1]   # z(1) = z1
        ])
        b_z = np.array([swing_start_point[2], z_apex, swing_to_point[2]])
        a_z, b_z, c_z = np.linalg.solve(A_z, b_z)

        # Parameterize t
        # t = np.linspace(0, 1, 100)

        # Generate 3D trajectory
        # Linear interpolation for x
        self.x_t = lambda t: (1 - t) * swing_start_point[0] + t * swing_to_point[0]
        self.y_t = lambda t: a_y * t**2 + b_y * t + c_y
        self.z_t = lambda t: a_z * t**2 + b_z * t + c_z
        
    def init_foot_trajectory_functions_motion(self,step_size_xy_mult, Frame_ID=10, DIR = 'N', q_start=None,debug = False):
        """
        Initialize the foot trajectory functions for x, y, and z over time t.

        Args:
            step_size_xy_mult (float): Multiplier for the step size in the XY plane.
            DIR (str): Direction of movement ('N', 'S', etc.).
            FOOT_ID (int): Frame ID of the foot.
        """
        # Generate direction vector based on specified direction (Gives the normal vector along which the pod moves)
        v = self.generate_direction_vector(DIR=DIR)
        # Get initial position of the foot in XY plane
        p1 = self.robot.framePlacement(self.qc, Frame_ID).translation
        # Compute the target position p2 in XY plane
        p2 = p1[0:2] + (self.HALF_STEPSIZE_XY* step_size_xy_mult * v)
        # Define trajectory functions for x(t) and y(t) as linear interpolation between p1 and p2
        p2 = np.append(p2, p1[2])

        y_clearance = 0.03           # Vertical clearance for y apex
        z_clearance = 0.01           # Lateral deviation for z apex

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
            
    def generate_joint_waypoints_stable(self, desired_pos, STEPS=5,Frame_ID=10, q_start=None):
        
        if q_start is None:
            q_start = self.qc

        if desired_pos == 1:
            self.init_foot_trajectory_functions_stable(Frame_ID, q_start=q_start)
        elif desired_pos ==2:
            self.trajectory_discretized (Frame_ID, q_start=q_start)
            
        timesteps =np.linspace(0,1,STEPS)
        # print(f'timesteps: {timesteps}')
        waypoints = [[round(self.x_t(t), 5), round(self.y_t(t), 5), round(self.z_t(t),5)] for t in timesteps]
        # print(f'waypoints: {waypoints}')
        
        # Convert waypoints to array for visualization
        points = np.array([waypoints[0], waypoints[-1]]).T
        
        # Visualize the foot trajectory if visualization is enabled
        if self.viz_flag:
            self.viz.viewer[('Wrist_' + str(Frame_ID) + '_trajectory')].set_object(
                g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0x00FFFF)))
            
        joint_traj = []
        q_prev = q_start
        for wp in waypoints:
            q_sol = self.inverse_geometry_stable(q_prev, Frame_ID, wp)
            joint_traj.append(q_sol)
            q_prev = q_sol  
        
        return joint_traj
    
    
    def generate_joint_waypoints_motion(self, step_size_xy_mult, STEPS=5, DIR='N', FOOT_ID=10):
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
        self.init_foot_trajectory_functions_motion(step_size_xy_mult=step_size_xy_mult, DIR=DIR, Frame_ID=FOOT_ID)
        # Create time steps from 0 to 1
        s = np.linspace(0, 1, STEPS)
        # Generate waypoints by evaluating trajectory functions at each time step
        waypoints = [[round(self.x_t(t), 5), round(self.y_t(t), 5), round(self.z_t(t), 5)] for t in s]
        # Convert waypoints to array for visualization
        points = np.array([waypoints[0], waypoints[-1]]).T
        # Visualize the foot trajectory if visualization is enabled
        if self.viz_flag:
            self.viz.viewer[('Foot_ID_' + str(FOOT_ID) + '_trajectory')].set_object(
                g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xffff00)))

        # Perform inverse kinematics to get joint configurations for each waypoint
        return [self.inverse_geometry_legmotion(self.qc, FOOT_ID, wp)
                for wp in waypoints]
    
    
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
        q_wps = self.generate_joint_waypoints_motion(step_size_xy_mult,DIR=DIR, FOOT_ID=self.FOOT_IDS[LEG], STEPS=STEPS)
        # self.plot_trajctory(title=f'Trajectory for Leg {LEG} after IK Before Interpolation', state=np.array(
        #     [self.forward_kinematics(q) for q in q_wps]))
        # print(self.qc)
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
            current_pos = self.robot.framePlacement(q_full, foot_id).translation
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

    def init_body_trajectory_functions(self, step_size_xy_mult, DIR='N'):

        # Generate direction vector based on specified direction
        v = self.generate_direction_vector(DIR=DIR)
        # Get initial position of the foot in XY plane
        p1 = self.robot.framePlacement(self.qc, self.BASE_FRAME_ID).translation
        # Compute the target position p2 in XY plane
        p2 = p1[0:2] + (self.HALF_STEPSIZE_XY * step_size_xy_mult * v)
        # Define trajectory functions for x(t) and y(t) as linear interpolation between p1 and p2
        p2 = np.append(p2, p1[2])
        self.x_t = lambda t: ((1 - t) * p1[0]) + (t * p2[0])
        self.y_t = lambda t:  ((1 - t) * p1[1]) + (t * p2[1])

        

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
        self.init_body_trajectory_functions(step_size_xy_mult=step_size_xy_mult, DIR=DIR)
        s = np.linspace(0, 1, STEPS)

        # Generate waypoints by evaluating trajectory functions at each time step
        waypoints = [np.concatenate(([round(self.x_t(t), 5), round(self.y_t(t), 5)], self.qc[2:7].copy())) for t in s]
        points = np.array(waypoints)[:, 0:3].T

        # Visualize the base trajectory if visualization is enabled
        if self.viz_flag:
            self.viz.viewer[('Base_trajectory')].set_object(g.Line(g.PointsGeometry(points), g.MeshBasicMaterial(color=0xffff00)))

        # Perform inverse kinematics to get joint configurations for each waypoint
        return [self.body_inverse_geometry(self.qc, wp) for wp in waypoints]

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
                q_t = self.compute_trajectory_p(q_wps[i], q_wps[i+1], t_init, t_goal, t)[0]
                q_traj = np.vstack((q_traj, q_t))
                t = (t + dt)
        # Remove the initial configuration
        return np.delete(q_traj, 0, axis=0)
    
    def compute_quad_homing(self):
        
        # Quad setting 
        k1 = self.generate_joint_waypoints_stable(desired_pos=1, STEPS=20, Frame_ID=10)
        k2 = self.generate_joint_waypoints_stable(desired_pos=1, STEPS=20, Frame_ID=18)
        k3 = self.generate_joint_waypoints_stable(desired_pos=1, STEPS=20, Frame_ID=34)
        k4 = self.generate_joint_waypoints_stable(desired_pos=1, STEPS=20, Frame_ID=42)
        for i in range(len(k1)):
         k1[i][16:19] = k3[i][16:19]
         
        for i in range(len(k2)):
         k2[i][19:22] = k4[i][19:22] 
         
        last_row_k1 = k1[-1][16:19]  # Extract cols 16,17,18 from last row of k1
        nxt_last_row_k1 = k1[-1][7:10]     

        for i in range(len(k2)):
            k2[i][16:19] = last_row_k1
            k2[i][7:10]  = nxt_last_row_k1     

        stack1 = np.vstack((k1, k2))
        
        kl1 = self.generate_joint_waypoints_stable(desired_pos=2, STEPS=20, Frame_ID=26)

        k = self.generate_joint_waypoints_stable(desired_pos=2, STEPS=20, Frame_ID=50)
        for i in range(len(k)):
         k[i][14:16] = kl1[i][14:16]

        last_row_stack1 = stack1[-1][7:14]  # Extract cols 16,17,18 from last row of k1
        nxt_last_row_stack1 = stack1[-1][16:23]
    
        for i in range(len(k)):
            k[i][7:14] = last_row_stack1
            k[i][16:23]  = nxt_last_row_stack1

        stack2 = np.vstack((stack1, k))
        self.homepose=stack2
        self.qc = stack2[-1]

        return stack2
    
    def compute_quad_gait(self,qset):
        
        
        step_size_xy_mult = 1
        t_goal = self.HALF_STEPSIZE_XY / 0.5
        STEPS = 5

        leg0_traj = self.generate_leg_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=0, t_goal=t_goal, STEPS=STEPS)
        leg3_traj = self.generate_leg_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=3, t_goal=t_goal, STEPS=STEPS)
        body_traj = self.generate_body_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', t_goal=t_goal, STEPS=STEPS)

        print(f"leg0_traj = {leg0_traj}")
        print(f"leg3_traj = {leg3_traj}")
        print(f"body_traj = {body_traj}")

        for i in range(len(leg0_traj)):
             leg0_traj[i][16:19] = leg3_traj[i][16:19]
             leg0_traj[i][0:7] = body_traj[i][0:7]
             leg0_traj[i][10:16] = body_traj[i][10:16]
             leg0_traj[i][19:] = body_traj[i][19:]
             leg0_traj[i][13:16] = qset[-1][13:16]
             leg0_traj[i][22:] = qset[-1][22:]

        qset = np.vstack((qset,leg0_traj))
        #_____________________________________________________________________________________________________________________________________________
        self.qc = qset[-1]
        STEP_CNT = 3
        step_size_xy_mult = 2
        for i in range(0, STEP_CNT):
            
            # Generate trajectories for legs 1, 3, and 5
            leg1_traj = self.generate_leg_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=1, t_goal=t_goal, STEPS=STEPS)
            print(f"leg1_traj = {leg1_traj}")
            leg4_traj = self.generate_leg_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=4, t_goal=t_goal, STEPS=STEPS)
            print(f"leg4_traj = {leg4_traj}")

            # Generate body trajectory
            body_traj = self.generate_body_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', t_goal=t_goal, STEPS=STEPS)
            print(f"body_traj = {body_traj}")
            
            for i in range(len(body_traj)):
                body_traj[i][10:13] =  leg1_traj[i][10:13]
                body_traj[i][19:22] =  leg4_traj[i][19:22]
                body_traj[i][13:16] =  qset[-1][13:16]
                body_traj[i][22:]   =  qset[-1][22:]


            # Append trajectories to q
            qset = np.vstack((qset,body_traj))
            # Update the current configuratselfn
            self.qc = qset[-1]
            # Generate trajectories for legs 0, 2, and 4 again
            leg0_traj = self.generate_leg_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=0, t_goal=t_goal, STEPS=STEPS)
            leg3_traj = self.generate_leg_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=3, t_goal=t_goal, STEPS=STEPS)

            # Generate body trajectory
            body_traj = self.generate_body_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', t_goal=t_goal, STEPS=STEPS)
            for i in range(len(body_traj)):
                body_traj[i][7:10] =  leg0_traj[i][7:10]
                body_traj[i][16:19] =  leg3_traj[i][16:19]
                body_traj[i][13:16] =  qset[-1][13:16]
                body_traj[i][22:]   =  qset[-1][22:]


            # Append trajectories to q
            qset = np.vstack((qset,body_traj))
            # Update the current configuratselfn
            self.qc = qset[-1]
        #_____________________________________________________________________________________________________________________________________________   
        step_size_xy_mult = 1
        
        leg1_traj = self.generate_leg_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=1, t_goal=t_goal, STEPS=STEPS)
        leg4_traj = self.generate_leg_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', LEG=4, t_goal=t_goal, STEPS=STEPS)
        body_traj = self.generate_body_joint_trajectory(step_size_xy_mult=step_size_xy_mult, DIR='N', t_goal=t_goal, STEPS=STEPS)

        for i in range(len(body_traj)):
                body_traj[i][10:13] =  leg1_traj[i][10:13]
                body_traj[i][19:22] =  leg4_traj[i][19:22]
                body_traj[i][13:16] =  qset[-1][13:16]
                body_traj[i][22:]   =  qset[-1][22:]

        qset = np.vstack((qset,body_traj))
        self.qc = qset[-1]       
        
        
        return qset
    
    def compute_quad_homing_disable(self):
        
        q_negate = 1*(self.homepose)
        q_dis = q_negate[::-1,:]
        for i in range(len(q_dis)):
            q_dis[i][0:7] = self.qc[0:7]
               
        return q_dis
        
    
    def animate_trajectory(self, joint_configs, delay=None):
      if not self.viz_flag:
          print("Visualization not enabled.")
          return

      for idx, q in enumerate(joint_configs):
          self.viz.display(q)
          time.sleep(delay)
    
    def save_file(self,a,b,c):
        
        gait_angles_file_path = Path(
        f'gait_angles_DIR_N_WP5_START_HALF_STEP__{strftime("%Y%m%d_%H%M%S")}.npy')
        gait_angles_file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(gait_angles_file_path, a)

        gait_angles_file_path = Path(
            f'gait_angles_DIR_N_WP5_MID_FULL_STEP__{strftime("%Y%m%d_%H%M%S")}.npy')
        gait_angles_file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(gait_angles_file_path, b)

        gait_angles_file_path = Path(
            f'gait_angles_DIR_N_WP5_END_HALF_STEP__{strftime("%Y%m%d_%H%M%S")}.npy')
        gait_angles_file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(gait_angles_file_path, c)
    
          
          
    
     
 
     

def main():
    
    home_visualizer = True
    debuging_mode = False
    IO = Hexapod(home_visualizer,debuging_mode)
   
    
    # time.sleep(5)
    qset = IO.compute_quad_homing()
    q= IO.compute_quad_gait(qset=qset)    
    print(q)
    disable = IO.compute_quad_homing_disable()

    Q_fin = np.vstack((q,disable))    
    IO.animate_trajectory(Q_fin, delay=0.01)
    
    a = Q_fin[0:75,:] 
    b = Q_fin[75:171,:]
    c = Q_fin[171:247,:]
    
    
    # IO.save_file(a,b,c)
    
    
 
    
    
    
if __name__ == "__main__":
    main()
   


