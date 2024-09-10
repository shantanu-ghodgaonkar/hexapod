import roboticstoolbox as rtb
# from roboticstoolbox.robot.DHRobot import DHRobot

class hexapod:
    
    def __init__(self):
        self.robot = rtb.robot.DHRobotDHRobot.URDF('urdf/hexy_test_5/hexy_test_5.urdf')

if __name__ == '__main__':
    hexy = hexapod()
    print('DEBUG POINT')