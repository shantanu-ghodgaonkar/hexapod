import pinocchio as pin
from sys import argv
import sys
from pathlib import Path
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from numpy.linalg import norm, solve
from time import sleep
import meshcat.geometry as g
import meshcat.transformations as tf
import math

PI = 3.141592
L1 = 47  # mm
L2 = 45.525  # mm
L3 = 94.6  # mm
LCO = 158.8  # mm
LOF = 0  # mm
D = 15  # mm : Step size


class model_loader:
    def __init__(self) -> None:
        # This path refers to pin source code but you can define your own directory here.
        # self.pin_model_dir = str(Path('hexy_urdf_v2_4_1_dae').absolute())
        self.pin_model_dir = str(
            Path('/home/shantanu/Documents/hexy_test_5').absolute())

        # You should change here to set up your own URDF file or just pass it as an argument of this example.
        # self.urdf_filename = (
        #     self.pin_model_dir
        #     + '/hexy_urdf_v2_4.urdf'
        # )

        self.urdf_filename = (
            self.pin_model_dir
            + '/hexy_test_5.urdf'
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

        pin.forwardKinematics(self.model, self.data, pin.neutral(self.model))
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(
            self.model, self.data, self.visual_model, self.visual_data)

        self.jointTs = np.zeros((6, 3, 3))
        for name, oMi in zip(self.model.names, self.data.oMi):
            if name != 'universe':
                self.jointTs[int(name[-1])-1][int(name[-2])] = oMi.translation

        self.visual_object_names = [
            geom.name for geom in self.visual_model.geometryObjects]
        self.feetTs = np.zeros((6, 3))
        for k, oMg in enumerate(self.visual_data.oMg):
            if ((self.visual_object_names[k][0] == 'T') & (self.visual_object_names[k][-1] == '1')):
                self.feetTs[int(self.visual_object_names[k]
                                [-3])-1] = oMg.translation

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
            print(("{:s}\t: {:d}\t: {: .2f} {: .2f} {: .2f}".format(self.visual_object_names[k],
                                                                    k, *oMg.translation.T.flat)))

        return q

    def XYZRPYtoSE3(xyzrpy):
        rotate = pin.utils.rotate
        R = rotate("x", xyzrpy[3]) @ rotate("y",
                                            xyzrpy[4]) @ rotate("z", xyzrpy[5])
        p = np.array(xyzrpy[:3])
        return pin.SE3(R, p)

    def inverse_kinematics(self, LEG=1, x=0.0, y=0.0, z=0.0):
        # gamma = math.atan()
        th1 = 0.0
        if x == 0.0:
            th1 = PI/2
        else:
            th1 = math.atan(y / x)
        c1 = math.cos(th1)
        s1 = math.sin(th1)

        th3 = math.asin(int((((((c1*x)+(s1*y) - L1) ** 2) + (z ** 2) -
                        (L3 ** 2) - (L2 ** 2)) / (2 * L2 * L3)) * 100)/100) - (PI/2)
        c3 = math.cos(th3)
        s3 = math.sin(th3)
        gamma = math.atan((L3*s3) / ((L3*c3) + L2))

        th2 = math.asin(
            z / (math.sqrt((((L3*c3) + L2) ** 2) + ((L3*s3) ** 2)))) - gamma

        print(f"Th1 = {th1}, Th2 = {th2}, Th3 = {th3}")

        return np.array([th1, th2, th3])

    def distance_between_points(self, point1, point2):
        """
        Calculate the Euclidean distance between two points in 3D space.

        :param point1: A tuple or list of (x1, y1, z1) coordinates.
        :param point2: A tuple or list of (x2, y2, z2) coordinates.
        :return: The Euclidean distance between the two points.
        """
        # Convert the points to numpy arrays for vectorized operations
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Calculate the distance using the Euclidean distance formula
        distance = np.sqrt(np.sum((point1 - point2) ** 2))

        return distance

    def inverse_kinematics_2(self, LEG=0, x0=0.0, y0=0.0, z0=0.0):
        L1 = self.distance_between_points(
            self.jointTs[LEG][1], self.jointTs[LEG][0])
        L2 = self.distance_between_points(
            self.jointTs[LEG][2], self.jointTs[LEG][1])
        L3 = self.distance_between_points(
            self.feetTs[LEG], self.jointTs[LEG][2])

        # θ1 = arctan(y0 / x0)
        theta1 = np.arctan2(y0, x0)

        # θ2 = arccos((-L3^2 + L2^2 + x0^2 + y0^2 + z0^2) / (2 * L2 * sqrt(x0^2 + y0^2 + z0^2))) + arctan(z0 / sqrt(x0^2 + y0^2))
        theta2 = np.arccos((-L3**2 + L2**2 + x0**2 + y0**2 + z0**2) / (2 * L2 *
                           np.sqrt(x0**2 + y0**2 + z0**2))) + np.arctan2(z0, np.sqrt(x0**2 + y0**2))

        # θ3 = - (π - β) = - arccos((x0^2 + y0^2 + z0^2 - L2^2 - L3^2) / (2 * L2 * L3))
        theta3 = - np.arccos((x0**2 + y0**2 + z0**2 - L2 **
                              2 - L3**2) / (2 * L2 * L3))

        return self.generate_config(LEG, theta1, theta2, theta3)

    def inverse_kinematics_iterative(self, JOINT_ID=6, oMdes=pin.SE3(np.eye(3), np.array([1.0, 0.0, 1.0]))):
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

    def create_axis_marker(self, viewer, direction, color):
        length = np.linalg.norm(direction)
        orientation = tf.rotation_matrix(
            np.arctan2(direction[1], direction[0]), [0, 0, 1])
        geometry = g.Cylinder(length, 0.005)
        material = g.MeshLambertMaterial(color=color, reflectivity=0.8)
        viewer.set_object(geometry, material)
        viewer.set_transform(
            orientation @ tf.translation_matrix([0, 0, length / 2]))

    # Function to plot the current locations of visual components
    def plot_visual_components(self, viz):
        for geom in self.visual_model.geometryObjects:
            frame_id = geom.parentFrame
            frame_name = self.model.frames[frame_id].name
            frame_placement = self.data.oMf[frame_id]

            # Create an axis marker for the frame
            axis_length = 0.1
            self.create_axis_marker(
                viz.viewer[frame_name + "_x"], [axis_length, 0, 0], [1, 0, 0])
            self.create_axis_marker(
                viz.viewer[frame_name + "_y"], [0, axis_length, 0], [0, 1, 0])
            self.create_axis_marker(
                viz.viewer[frame_name + "_z"], [0, 0, axis_length], [0, 0, 1])

            # Set the transformation of the axis marker to match the frame
            T = tf.translation_matrix(frame_placement.translation)
            R = frame_placement.rotation
            T[:3, :3] = R

            # Add the axis marker to the visualizer
            viz.viewer[frame_name].set_transform(T)

    def move_forward_inf(self, viz):
        # # HOME CONFIG
        # q0 = np.zeros(18)
        # viz.display(q0)
        # sleep(1.5)

        # # L135 : UP
        # # L246 : DN
        # q1_1 = np.array([0, 0, 0])
        # q2_1 = np.array([0, 0, 0])
        # q3_1 = np.array([0, -PI/9, 0])
        # q4_1 = np.array([0, 0, 0])
        # q5_1 = np.array([0, PI/9, 0])
        # q6_1 = np.array([0, 0, 0])
        # q_fin_1 = np.concatenate((q1_1, q2_1, q3_1, q4_1, q5_1, q6_1))
        # viz.display(q_fin_1)
        # sleep(1.5)

        # # L135 : UP + FORW
        # # L246 : DN
        # q1_2 = np.array([0, 0, 0])
        # q2_2 = np.array([0, 0, 0])
        # q3_2 = np.array([PI/6, -PI/9, 0])
        # q4_2 = np.array([0, 0, 0])
        # q5_2 = np.array([-PI/6, PI/9, 0])
        # q6_2 = np.array([0, 0, 0])
        # q_fin_2 = np.concatenate((q1_2, q2_2, q3_2, q4_2, q5_2, q6_2))
        # viz.display(q_fin_2)
        # sleep(1.5)

        # # L135 : DN + FORW
        # # L246 : DN
        # q1_3 = np.array([0, 0, 0])
        # q2_3 = np.array([0, 0, 0])
        # q3_3 = np.array([PI/6, 0, 0])
        # q4_3 = np.array([0, 0, 0])
        # q5_3 = np.array([-PI/6, 0, 0])
        # q6_3 = np.array([0, 0, 0])
        # q_fin_3 = np.concatenate((q1_3, q2_3, q3_3, q4_3, q5_3, q6_3))
        # viz.display(q_fin_3)
        # sleep(1.5)

        # # L135 : DN + FORW
        # # L246 : UP
        # q1_4 = np.array([0, 0, 0])
        # q2_4 = np.array([0, 0, 0])
        # q3_4 = np.array([PI/6, 0, 0])
        # q4_4 = np.array([0, PI/9, 0])
        # q5_4 = np.array([-PI/6, 0, 0])
        # q6_4 = np.array([0, PI/9, 0])
        # q_fin_4 = np.concatenate((q1_4, q2_4, q3_4, q4_4, q5_4, q6_4))
        # viz.display(q_fin_4)
        # sleep(1.5)

        # while (1):
        #     # L135 : DN + BACK
        #     # L246 : UP + FORW
        #     q1_7 = np.array([PI/6, 0, 0])
        #     q2_7 = np.array([0, -PI/9, 0])
        #     q3_7 = np.array([-PI/6, 0, 0])
        #     q4_7 = np.array([PI/6, PI/9, 0])
        #     q5_7 = np.array([PI/6, 0, 0])
        #     q6_7 = np.array([-PI/6, PI/9, 0])
        #     q_fin_7 = np.concatenate((q1_7, q2_7, q3_7, q4_7, q5_7, q6_7))
        #     viz.display(q_fin_7)
        #     sleep(1.5)

        #     # L135 : DN + BACK
        #     # L246 : DN + FORW
        #     q1_8 = np.array([PI/6, 0, 0])
        #     q2_8 = np.array([0, 0, 0])
        #     q3_8 = np.array([-PI/6, 0, 0])
        #     q4_8 = np.array([PI/6, 0, 0])
        #     q5_8 = np.array([PI/6, 0, 0])
        #     q6_8 = np.array([-PI/6, 0, 0])
        #     q_fin_8 = np.concatenate((q1_8, q2_8, q3_8, q4_8, q5_8, q6_8))
        #     viz.display(q_fin_8)
        #     sleep(1.5)

        #     # L135 : UP + FORW
        #     # L246 : DN + BACK
        #     q1_8 = np.array([0, -PI/9, 0])
        #     q2_8 = np.array([-PI/6, 0, 0])
        #     q3_8 = np.array([PI/6, -PI/9, 0])
        #     q4_8 = np.array([-PI/6, 0, 0])
        #     q5_8 = np.array([-PI/6, PI/9, 0])
        #     q6_8 = np.array([PI/6, 0, 0])
        #     q_fin_8 = np.concatenate((q1_8, q2_8, q3_8, q4_8, q5_8, q6_8))
        #     viz.display(q_fin_8)
        #     sleep(1.5)

        #     # L135 : DN + FORW
        #     # L246 : DN + BACK
        #     q1_9 = np.array([0, 0, 0])
        #     q2_9 = np.array([-PI/6, 0, 0])
        #     q3_9 = np.array([PI/6, 0, 0])
        #     q4_9 = np.array([-PI/6, 0, 0])
        #     q5_9 = np.array([-PI/6, 0, 0])
        #     q6_9 = np.array([PI/6, 0, 0])
        #     q_fin_9 = np.concatenate((q1_9, q2_9, q3_9, q4_9, q5_9, q6_9))
        #     viz.display(q_fin_9)
        #     sleep(1.5)
        q1 = self.generate_config(leg=0, q1=PI/9, q2=-PI/9, q3=0) + \
            self.generate_config(leg=1, q1=0, q2=0, q3=0) + \
            self.generate_config(leg=2, q1=-PI/9, q2=PI/9, q3=0) + \
            self.generate_config(leg=1, q1=0, q2=0, q3=0) + \
            self.generate_config(leg=4, q1=PI/9, q2=-PI/9, q3=0) +\
            self.generate_config(leg=5, q1=0, q2=0, q3=0)

        q2 = self.generate_config(leg=0, q1=PI/9, q2=0, q3=0) + \
            self.generate_config(leg=1, q1=0, q2=0, q3=0) + \
            self.generate_config(leg=2, q1=-PI/9, q2=0, q3=0) + \
            self.generate_config(leg=3, q1=0, q2=0, q3=0) + \
            self.generate_config(leg=4, q1=PI/9, q2=0, q3=0) +\
            self.generate_config(leg=5, q1=0, q2=0, q3=0)

        q3 = self.generate_config(leg=0, q1=PI/9, q2=0, q3=0) + \
            self.generate_config(leg=1, q1=0, q2=PI/9, q3=0) + \
            self.generate_config(leg=2, q1=-PI/9, q2=0, q3=0) + \
            self.generate_config(leg=3, q1=0, q2=PI/9, q3=0) + \
            self.generate_config(leg=4, q1=PI/9, q2=0, q3=0) +\
            self.generate_config(leg=5, q1=-PI/3, q2=-PI/9, q3=0)

        q4 = self.generate_config(leg=0, q1=PI/6, q2=-PI/9, q3=0) + \
            self.generate_config(leg=1, q1=0, q2=0, q3=0) + \
            self.generate_config(leg=2, q1=-PI/12, q2=PI/9, q3=0) + \
            self.generate_config(leg=1, q1=0, q2=0, q3=0) + \
            self.generate_config(leg=4, q1=PI/6, q2=-PI/9, q3=0) +\
            self.generate_config(leg=5, q1=0, q2=0, q3=0)

        # while (1):
        #     viz.display(q1)
        #     sleep(2)
        #     # viz.display(q2)
        #     # sleep(1)
        #     # viz.display(q3)
        #     viz.display(q4)
        #     sleep(1)

    def generate_config(self, leg=0, q1=0, q2=0, q3=0):
        return np.concatenate((np.zeros(leg*3), [q1, q2, q3], np.zeros((5-leg)*3)))

    def get_direction_slope(self, edgeNum=0):
        match edgeNum:
            case 0:
                revJnt1 = self.data.oMi.__getitem__(1).translation
                revJnt2 = self.data.oMi.__getitem__(4).translation
            case 1:
                revJnt1 = self.data.oMi.__getitem__(4).translation
                revJnt2 = self.data.oMi.__getitem__(7).translation
            case 2:
                revJnt1 = self.data.oMi.__getitem__(7).translation
                revJnt2 = self.data.oMi.__getitem__(10).translation
            case 3:
                revJnt1 = self.data.oMi.__getitem__(10).translation
                revJnt2 = self.data.oMi.__getitem__(13).translation
            case 4:
                revJnt1 = self.data.oMi.__getitem__(13).translation
                revJnt2 = self.data.oMi.__getitem__(16).translation
            case 5:
                revJnt1 = self.data.oMi.__getitem__(16).translation
                revJnt2 = self.data.oMi.__getitem__(1).translation
            case _:
                revJnt1 = self.data.oMi.__getitem__(1).translation
                revJnt2 = self.data.oMi.__getitem__(4).translation

        return -1/((revJnt2[1] - revJnt1[1]) / (revJnt2[0] - revJnt1[0]))

    def get_forward_points(self):
        # Get the slope for the line depicting the direction of motion
        m = self.get_direction_slope(0)

        # Obtain current foot positions
        f1c = self.visual_data.oMg.__getitem__(4).translation
        f2c = self.visual_data.oMg.__getitem__(8).translation
        f3c = self.visual_data.oMg.__getitem__(12).translation
        f4c = self.visual_data.oMg.__getitem__(16).translation
        f5c = self.visual_data.oMg.__getitem__(20).translation
        f6c = self.visual_data.oMg.__getitem__(24).translation

        # Compute the waypoints
        f1n = np.array([
            (f1c[0] + (D/math.sqrt(1 + (m**2)))),  (f1c[1] +
                                                    (m * (f1c[0] + (D/math.sqrt(1 + (m**2)))))), f1c[2]
        ])

        f2n = np.array([
            (f2c[0] + (D/math.sqrt(1 + (m**2)))),  (f2c[1] +
                                                    (m * (f2c[0] + (D/math.sqrt(1 + (m**2)))))), f2c[2]
        ])

        f3n = np.array([
            (f3c[0] + (D/math.sqrt(1 + (m**2)))),  (f3c[1] +
                                                    (m * (f3c[0] + (D/math.sqrt(1 + (m**2)))))), f3c[2]
        ])

        f4n = np.array([
            (f4c[0] + (D/math.sqrt(1 + (m**2)))),  (f4c[1] +
                                                    (m * (f4c[0] + (D/math.sqrt(1 + (m**2)))))), f4c[2]
        ])

        f5n = np.array([
            (f5c[0] + (D/math.sqrt(1 + (m**2)))),  (f5c[1] +
                                                    (m * (f5c[0] + (D/math.sqrt(1 + (m**2)))))), f5c[2]
        ])

        f6n = np.array([
            (f6c[0] + (D/math.sqrt(1 + (m**2)))),  (f6c[1] +
                                                    (m * (f6c[0] + (D/math.sqrt(1 + (m**2)))))), f6c[2]
        ])

        return f1n, f2n, f3n, f4n, f5n, f6n


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
    viz.displayCollisions(visibility=False)

    # model.move_forward_inf(viz)

    viz.display(pin.neutral(model.__getattribute__('model')))
    model.forward_kinematics(pin.neutral(model.__getattribute__('model')))

    # f1n, f2n, f3n, f4n, f5n, f6n = model.get_forward_points()
    # q1 = model.inverse_kinematics_iterative(
    #     4, pin.SE3(np.eye(3), f1n))
    # q3 = model.inverse_kinematics_iterative(
    #     4, pin.SE3(np.eye(3), f3n))
    # q5 = model.inverse_kinematics_iterative(
    #     4, pin.SE3(np.eye(3), f5n))

    # viz.display((q1+q3+q5))
    # model.move_forward_inf(viz)

    # Define a point in 3D space
    # point = np.array([0.5, 0.5, 0.5])  # Example coordinates

    # # Plot the point in MeshCat
    # viz.viewer["point"].set_object(
    #     g.Sphere(0.01), g.MeshLambertMaterial(color=0xff0000))
    # viz.viewer["point"].set_transform(tf.translation_matrix(point))

    # model.visualize_visual_frames(viz)
    # sleep(3)

    # q1 = model.inverse_kinematics_iterative(
    #     4, pin.SE3(np.eye(3), np.array([-0.02, - 1.38,  0.01])))
    # q2_raw = model.inverse_kinematics(x=-0.02, y=-1.38,  z=0.01)
    # q2 = model.generate_config(0, q2_raw[0], q2_raw[1], q2_raw[2])
    # q2 = model.inverse_kinematics_2(LEG=0, x0=-0.02, y0=-1.38,  z0=0.1)
    # model.forward_kinematics(q1)
    # sleep(3)
    # model.forward_kinematics(q2)
    # viz.display(q2)
    # viz.display(np.concatenate((q, np.zeros(15))))
