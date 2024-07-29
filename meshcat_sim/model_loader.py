import pinocchio as pin
from sys import argv
import sys
from pathlib import Path
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from numpy.linalg import norm, solve
from time import sleep

PI = 3.141592


class model_loader:
    def __init__(self) -> None:
        # This path refers to pin source code but you can define your own directory here.
        self.pin_model_dir = str(Path('hexy_urdf_v2_4_1_dae').absolute())

        # You should change here to set up your own URDF file or just pass it as an argument of this example.
        self.urdf_filename = (
            self.pin_model_dir
            + '/hexy_urdf_v2_4.urdf'
        )

        # Load the urdf model
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            self.urdf_filename, geometry_types=[pin.GeometryType.COLLISION, pin.GeometryType.VISUAL])
        print("model name: " + self.model.name)

        # Add collisition pairs
        self.collision_model.addAllCollisionPairs()
        print("num collision pairs - initial:",
              len(self.collision_model.collisionPairs))

        # Print some information about the model
        # for name, function in self.model.__class__.__dict__.items():
        #     print(' **** %s: %s' % (name, function.__doc__))

        # Create data required by the algorithms
        self.data, self.collision_data, self.visual_data = pin.createDatas(
            self.model, self.collision_model, self.visual_model)

    def random_config(self):
        # Sample a random configuration
        q = pin.randomConfiguration(self.model)
        print("q: %s" % q.T)
        return q

    def compute_collisions(self, q):
        # Compute all the collisions
        pin.computeCollisions(
            self.model, self.data, self.collision_model, self.collision_data, q, False)

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
        pin.updateGeometryPlacements(
            self.model, self.data, self.collision_model, self.collision_data)
        pin.updateGeometryPlacements(
            self.model, self.data, self.visual_model, self.visual_data)

        pin.updateFramePlacements(self.model, self.data)

        # Print out the placement of each joint of the kinematic tree
        print("\nJoint placements:")
        for name, oMi in zip(self.model.names, self.data.oMi):
            print(("{:<24} : {: .2f} {: .2f} {: .2f}".format(
                name, *oMi.translation.T.flat)))

        # Print out the placement of each collision geometry object
        print("\nCollision object placements:")
        for k, oMg in enumerate(self.collision_data.oMg):
            print(("{:d} : {: .2f} {: .2f} {: .2f}".format(
                k, *oMg.translation.T.flat)))

        # Print out the placement of each visual geometry object
        print("\nVisual object placements:")
        for k, oMg in enumerate(self.visual_data.oMg):
            print(("{:d} : {: .2f} {: .2f} {: .2f}".format(
                k, *oMg.translation.T.flat)))

        return q

    def XYZRPYtoSE3(xyzrpy):
        rotate = pin.utils.rotate
        R = rotate("x", xyzrpy[3]) @ rotate("y",
                                            xyzrpy[4]) @ rotate("z", xyzrpy[5])
        p = np.array(xyzrpy[:3])
        return pin.SE3(R, p)

    def inverse_kinematics(self, JOINT_ID=6, oMdes=pin.SE3(np.eye(3), np.array([1.0, 0.0, 1.0]))):
        q = pin.neutral(self.model)
        eps = 1e-4
        IT_MAX = 1000
        DT = 1e-1
        damp = 1e-12

        i = 0
        while True:
            pin.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[JOINT_ID].actInv(oMdes)
            err = pin.log(iMd).vector  # in joint frame
            if norm(err) < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break
            J = pin.computeJointJacobian(
                self.model, self.data, q, JOINT_ID)  # in joint frame
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * DT)
            if not i % 10:
                print("%d: error = %s" % (i, err.T))
            i += 1

        if success:
            print("Convergence achieved!")
        else:
            print(
                "\nWarning: the iterative algorithm has not reached convergence to the desired precision"
            )

        print("\nresult: %s" % q.flatten().tolist())
        print("\nfinal error: %s" % err.T)
        return np.array(q.flatten().tolist())

    def move_forward_inf(self, viz):
        # HOME CONFIG
        q0 = np.zeros(18)
        viz.display(q0)
        sleep(1.5)

        # L135 : UP
        # L246 : DN
        q1_1 = np.array([0, 0, 0])
        q2_1 = np.array([0, 0, 0])
        q3_1 = np.array([0, -PI/9, 0])
        q4_1 = np.array([0, 0, 0])
        q5_1 = np.array([0, PI/9, 0])
        q6_1 = np.array([0, 0, 0])
        q_fin_1 = np.concatenate((q1_1, q2_1, q3_1, q4_1, q5_1, q6_1))
        viz.display(q_fin_1)
        sleep(1.5)

        # L135 : UP + FORW
        # L246 : DN
        q1_2 = np.array([0, 0, 0])
        q2_2 = np.array([0, 0, 0])
        q3_2 = np.array([PI/6, -PI/9, 0])
        q4_2 = np.array([0, 0, 0])
        q5_2 = np.array([-PI/6, PI/9, 0])
        q6_2 = np.array([0, 0, 0])
        q_fin_2 = np.concatenate((q1_2, q2_2, q3_2, q4_2, q5_2, q6_2))
        viz.display(q_fin_2)
        sleep(1.5)

        # L135 : DN + FORW
        # L246 : DN
        q1_3 = np.array([0, 0, 0])
        q2_3 = np.array([0, 0, 0])
        q3_3 = np.array([PI/6, 0, 0])
        q4_3 = np.array([0, 0, 0])
        q5_3 = np.array([-PI/6, 0, 0])
        q6_3 = np.array([0, 0, 0])
        q_fin_3 = np.concatenate((q1_3, q2_3, q3_3, q4_3, q5_3, q6_3))
        viz.display(q_fin_3)
        sleep(1.5)

        # L135 : DN + FORW
        # L246 : UP
        q1_4 = np.array([0, 0, 0])
        q2_4 = np.array([0, 0, 0])
        q3_4 = np.array([PI/6, 0, 0])
        q4_4 = np.array([0, PI/9, 0])
        q5_4 = np.array([-PI/6, 0, 0])
        q6_4 = np.array([0, PI/9, 0])
        q_fin_4 = np.concatenate((q1_4, q2_4, q3_4, q4_4, q5_4, q6_4))
        viz.display(q_fin_4)
        sleep(1.5)

        while (1):
            # L135 : DN + BACK
            # L246 : UP + FORW
            q1_7 = np.array([PI/6, 0, 0])
            q2_7 = np.array([0, -PI/9, 0])
            q3_7 = np.array([-PI/6, 0, 0])
            q4_7 = np.array([PI/6, PI/9, 0])
            q5_7 = np.array([PI/6, 0, 0])
            q6_7 = np.array([-PI/6, PI/9, 0])
            q_fin_7 = np.concatenate((q1_7, q2_7, q3_7, q4_7, q5_7, q6_7))
            viz.display(q_fin_7)
            sleep(1.5)

            # L135 : DN + BACK
            # L246 : DN + FORW
            q1_8 = np.array([PI/6, 0, 0])
            q2_8 = np.array([0, 0, 0])
            q3_8 = np.array([-PI/6, 0, 0])
            q4_8 = np.array([PI/6, 0, 0])
            q5_8 = np.array([PI/6, 0, 0])
            q6_8 = np.array([-PI/6, 0, 0])
            q_fin_8 = np.concatenate((q1_8, q2_8, q3_8, q4_8, q5_8, q6_8))
            viz.display(q_fin_8)
            sleep(1.5)

            # L135 : UP + FORW
            # L246 : DN + BACK
            q1_8 = np.array([0, -PI/9, 0])
            q2_8 = np.array([-PI/6, 0, 0])
            q3_8 = np.array([PI/6, -PI/9, 0])
            q4_8 = np.array([-PI/6, 0, 0])
            q5_8 = np.array([-PI/6, PI/9, 0])
            q6_8 = np.array([PI/6, 0, 0])
            q_fin_8 = np.concatenate((q1_8, q2_8, q3_8, q4_8, q5_8, q6_8))
            viz.display(q_fin_8)
            sleep(1.5)

            # L135 : DN + FORW
            # L246 : DN + BACK
            q1_9 = np.array([0, 0, 0])
            q2_9 = np.array([-PI/6, 0, 0])
            q3_9 = np.array([PI/6, 0, 0])
            q4_9 = np.array([-PI/6, 0, 0])
            q5_9 = np.array([-PI/6, 0, 0])
            q6_9 = np.array([PI/6, 0, 0])
            q_fin_9 = np.concatenate((q1_9, q2_9, q3_9, q4_9, q5_9, q6_9))
            viz.display(q_fin_9)
            sleep(1.5)


if __name__ == '__main__':
    model = model_loader()

    # Start a new MeshCat server and client.
    # Note: the server can also be started separately using the "meshcat-server" command in a terminal:
    # this enables the server to remain active after the current script ends.
    #
    # Option open=True pens the visualizer.
    # Note: the visualizer can also be opened seperately by visiting the provided URL.
    try:
        viz = MeshcatVisualizer(model.__getattribute__('model'), model.__getattribute__(
            'collision_model'), model.__getattribute__('visual_model'))
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

    # model.move_forward_inf(viz)

    viz.display(pin.neutral(model.__getattribute__('model')))

    sleep(3)

    viz.display(model.inverse_kinematics(
        4, pin.SE3(np.eye(3), np.array([0.27, -1.38, 0.01]))))
