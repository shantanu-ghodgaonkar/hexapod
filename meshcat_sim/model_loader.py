import pinocchio as pin
from sys import argv
import sys
from pathlib import Path
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
 
 
class model_loader :
    def __init__(self) -> None:
        # This path refers to pin source code but you can define your own directory here.
        # self.pin_model_dir = str(Path('hexy_urdf_v2_dae').absolute())
        self.pin_model_dir = str(Path('hexy_urdf_v2_4_1_dae').absolute())
    
        # You should change here to set up your own URDF file or just pass it as an argument of this example.
        # self.urdf_filename = (
        #     self.pin_model_dir
        #     + '/hexy_urdf_v2.urdf'
        # )
        self.urdf_filename = (
            self.pin_model_dir
            + '/hexy_urdf_v2_4.urdf'
        )
        
        # Load the urdf model
        self.model, self.collision_model, self.visual_model  = pin.buildModelsFromUrdf(self.urdf_filename, geometry_types=[pin.GeometryType.COLLISION,pin.GeometryType.VISUAL])
        print("model name: " + self.model.name)
        
        # Add collisition pairs
        self.collision_model.addAllCollisionPairs()
        print("num collision pairs - initial:", len(self.collision_model.collisionPairs))
        
        # Print some information about the model
        # for name, function in self.model.__class__.__dict__.items():
        #     print(' **** %s: %s' % (name, function.__doc__))
        
        # Create data required by the algorithms
        self.data, self.collision_data, self.visual_data = pin.createDatas(self.model, self.collision_model, self.visual_model)
        
           
    def random_config(self):
        # Sample a random configuration
        q = pin.randomConfiguration(self.model)
        print("q: %s" % q.T)
        return q
    
    def compute_collisions(self, q):
        # Compute all the collisions
        pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, q, False)
 
        # Print the status of collision for all collision pairs
        for k in range(len(self.collision_model.collisionPairs)):
            cr = self.collision_data.collisionResults[k]
            cp = self.collision_model.collisionPairs[k]
            print(
                "collision pair:",
                cp.first,
                ",",
                cp.second,
                "- collision:",
                "Yes" if cr.isCollision() else "No",
            )
    
    def forward_kinematics(self, q):
        # Perform the forward kinematics over the kinematic tree
        pin.forwardKinematics(self.model, self.data, q)


        # Compute all the collisions
        self.compute_collisions(q)


        # Update Geometry models
        pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data)
        pin.updateGeometryPlacements(self.model, self.data, self.visual_model, self.visual_data)
        
        # pin.updateFramePlacements(self.model, self.data)
        
        # Print out the placement of each joint of the kinematic tree
        print("\nJoint placements:")
        for name, oMi in zip(self.model.names, self.data.oMi):
            print(("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat)))
        
        # Print out the placement of each collision geometry object
        print("\nCollision object placements:")
        for k, oMg in enumerate(self.collision_data.oMg):
            print(("{:d} : {: .2f} {: .2f} {: .2f}".format(k, *oMg.translation.T.flat)))
        
        # Print out the placement of each visual geometry object
        print("\nVisual object placements:")
        for k, oMg in enumerate(self.visual_data.oMg):
            print(("{:d} : {: .2f} {: .2f} {: .2f}".format(k, *oMg.translation.T.flat)))
            
        return q
     
    def XYZRPYtoSE3(xyzrpy):
        rotate = pin.utils.rotate
        R = rotate("x", xyzrpy[3]) @ rotate("y", xyzrpy[4]) @ rotate("z", xyzrpy[5])
        p = np.array(xyzrpy[:3])
        return pin.SE3(R, p)
    
    


            
            
            
            
            
if __name__ == '__main__':
    model = model_loader()
    
    
    # Start a new MeshCat server and client.
    # Note: the server can also be started separately using the "meshcat-server" command in a terminal:
    # this enables the server to remain active after the current script ends.
    #
    # Option open=True pens the visualizer.
    # Note: the visualizer can also be opened seperately by visiting the provided URL.
    try:
        viz = MeshcatVisualizer(model.__getattribute__('model'), model.__getattribute__('collision_model'), model.__getattribute__('visual_model'))
        viz.initViewer(open=True)
        # model.apply_colors(viz)
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install Python meshcat"
        )
        print(err)
        sys.exit(0)
    
    # Load the robot in the viewer.
    viz.loadViewerModel()
    
    viz.displayFrames(visibility=True)
    viz.displayCollisions(visibility=True)
    # viz.displayVisuals(visibility=True)
    
    # Display a robot random configuration.
    # q0 = model.random_config()
    # viz.display(q0)
    # model.compute_collisions(q0)
    # viz.displayVisuals(True)
    
    import numpy as np
    import math
    from time import sleep
    
    # for i in range(0,30):
    #     sleep(1)
    #     viz.display(np.zeros(18))
    #     sleep(1)
    #     viz.display(np.ones(18))  
    
    q0 = np.zeros(18)
    q1_1 = np.array([-math.pi/3, -math.pi/8, 0])
    # q2_0 = np.append(np.zeros(3), np.append([0, -math.pi/4, 0], np.zeros(12)))
    # q2_1 = np.append(np.zeros(3), np.append([0, -math.pi/4, math.pi/2], np.zeros(12)))
    q3_1 = np.array([math.pi/3, -math.pi/8, 0])
    q5_1 = np.array([0, math.pi/6, 0])
    q5_2 = np.array([0, math.pi/6, -math.pi/2])
    q_fin =np.concatenate((q1_1, np.zeros(3), q3_1, np.zeros(3), q5_2, np.zeros(3))) 
    viz.display(q_fin)
    
    
    