import pinocchio
from sys import argv
import sys
from pathlib import Path
from pinocchio.visualize import MeshcatVisualizer
 
 
class model_loader :
    def __init__(self) -> None:
        # This path refers to Pinocchio source code but you can define your own directory here.
        self.pinocchio_model_dir = str(Path('hexy_urdf_v2_dae').absolute())
    
        # You should change here to set up your own URDF file or just pass it as an argument of this example.
        self.urdf_filename = (
            self.pinocchio_model_dir
            + '/hexy_urdf_v2.urdf'
        )
        
        # Load the urdf model
        self.model, self.collision_model, self.visual_model  = pinocchio.buildModelsFromUrdf(self.urdf_filename)
        print("model name: " + self.model.name)
        
        # Create data required by the algorithms
        self.data, self.collision_data, self.visual_data = pinocchio.createDatas(self.model, self.collision_model, self.visual_model)
        
    def get_model_dir(self):
        return self.pinocchio_model_dir
    
    def get_urdf_path(self):
        return self.model
    
    def random_config(self):
        # Sample a random configuration
        q = pinocchio.randomConfiguration(self.model)
        print("q: %s" % q.T)
        return q
    
    def forward_kinematics(self, q):
        # Perform the forward kinematics over the kinematic tree
        pinocchio.forwardKinematics(self.model, self.data, q)

        # Update Geometry models
        pinocchio.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data)
        pinocchio.updateGeometryPlacements(self.model, self.data, self.visual_model, self.visual_data)
        
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
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install Python meshcat"
        )
        print(err)
        sys.exit(0)
    
    # Load the robot in the viewer.
    viz.loadViewerModel()
    
    # Display a robot random configuration.
    q0 = model.random_config()
    viz.display(q0)
    viz.displayVisuals(True)