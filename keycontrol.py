import os, sys, ctypes, time
import dynamixel_sdk as dxl 
import numpy as np
import keyboard as kb

class HexapodController:
    def __init__(self, device_name, baudrate, gait_file):
        self.device_name = device_name
        self.baudrate = baudrate
        self.gait_file = gait_file
        self.port = dxl.PortHandler(self.device_name)
        self.packet = dxl.PacketHandler(2.0)
        self.group_sync_write = None
        self.gait_angles = None
        self.action = None
        self.load_gait_file()
        self.setup_motors()

    def load_gait_file(self):
        try:
            self.gait_angles = np.load(self.gait_file, allow_pickle=True)
            print(f"Loaded gait file with {self.gait_angles.shape[0]} frames")
        except Exception as e:
            print(f"Error loading gait file: {e}")

    def setup_motors(self):
        self.group_sync_write = dxl.GroupSyncWrite(self.port, self.packet, 116, 4)
        try:
            self.packet.write1ByteTxOnly(self.port, 254, 65, 1)
            self.packet.write1ByteTxOnly(self.port, 254, 64, 1)
            self.packet.write2ByteTxOnly(self.port, 254, 84, 800)  # P gain
            self.packet.write2ByteTxOnly(self.port, 254, 82, 0)    # I gain
            self.packet.write2ByteTxOnly(self.port, 254, 80, 64)   # D gain
            time.sleep(0.2)
        except Exception as e:
            print(f"Error setting up motors: {e}")

    def walk(self, seq, steps=2, delay=0.02):
        if self.action != 1:
            self.setup_motors()
        self.action = 1

        try:
            print(f"Starting walking")
            for step in range(steps):
                print(f"Step {step + 1}/{steps}")
                for frame in range(self.gait_angles.shape[0]):
                    if not kb.is_pressed('up') and not kb.is_pressed('down') and not kb.is_pressed('left') and not kb.is_pressed('right'):
                        print("Key released, stopping movement")
                        time.sleep(0.1)
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        time.sleep(0.1)
                        return

                    time.sleep(0.04)
                    self.group_sync_write.clearParam()

                    for motor_id in range(0, 18, 1):
                        angle = int(self.gait_angles[frame, motor_id])
                        angle = max(0, min(4095, angle))

                        param = [
                            dxl.DXL_LOBYTE(dxl.DXL_LOWORD(angle)),
                            dxl.DXL_HIBYTE(dxl.DXL_LOWORD(angle)),
                            dxl.DXL_LOBYTE(dxl.DXL_HIWORD(angle)),
                            dxl.DXL_HIBYTE(dxl.DXL_HIWORD(angle))
                        ]
                        self.group_sync_write.addParam(seq[motor_id], param)
                    print(f"Frame Complete {frame}")
                    result = self.group_sync_write.txPacket()

                time.sleep(delay)
                self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                time.sleep(delay)

            print("Walking completed")

        except Exception as e:
            print(f"Error during walking: {e}")

        finally:
            self.group_sync_write.clearParam()
            self.packet.write1ByteTxOnly(self.port, 254, 65, 0)
            print("Motors disabled")

    def move_forward(self):
        seq = (4, 10, 16, 5, 11, 17, 6, 12, 18, 1, 7, 13, 2, 8, 14, 3, 9, 15)
        self.walk(seq)

    def move_backward(self):
        seq = (1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 5, 11, 17, 6, 12, 18)
        self.walk(seq)

    def rotate_left(self):
        p1 = (7, 9, 11)
        p2 = (8, 10, 12)
        d = 0.02
        d2 = 0.5
        if self.action != 5:
            self.action = 5
            time.sleep(0.2)
        try:
            while kb.is_pressed('left'):
                for i in p1:
                    if not kb.is_pressed('left'):
                        print("Key released, stopping turn left action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 1500)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 1500)
                    time.sleep(d)
                time.sleep(0.5)
                for i in p2:
                    if not kb.is_pressed('left'):
                        print("Key released, stopping turn left action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i - 6, 116, 1550)
                    time.sleep(d)
                time.sleep(d2)
                for i in p1:
                    if not kb.is_pressed('left'):
                        print("Key released, stopping turn left action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 2048)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 2048)
                    time.sleep(d)
                time.sleep(0.5)
                time.sleep(d2)
                for i in p2:
                    if not kb.is_pressed('left'):
                        print("Key released, stopping turn left action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 1500)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 1500)
                    time.sleep(d)
                time.sleep(0.5)
                for i in p2:
                    if not kb.is_pressed('left'):
                        print("Key released, stopping turn left action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i - 6, 116, 2048)
                    time.sleep(d)
                time.sleep(d2)
                self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                time.sleep(d2)
        except KeyboardInterrupt:
            print("Turn left action interrupted")
        except Exception as e:
            print(f"Error during turn left action: {e}")

    def rotate_right(self):
        p1 = (7, 9, 11)
        p2 = (8, 10, 12)
        d = 0.02
        d2 = 0.5
        if self.action != 6:
            self.action = 6
            time.sleep(0.2)
        try:
            while kb.is_pressed('right'):
                for i in p1:
                    if not kb.is_pressed('right'):
                        print("Key released, stopping turn right action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 1500)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 1500)
                    time.sleep(d)
                time.sleep(0.5)
                for i in p2:
                    if not kb.is_pressed('right'):
                        print("Key released, stopping turn right action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i - 6, 116, 2600)
                    time.sleep(d)
                time.sleep(d2)
                for i in p1:
                    if not kb.is_pressed('right'):
                        print("Key released, stopping turn right action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 2048)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 2048)
                    time.sleep(d)
                time.sleep(0.5)
                time.sleep(d2)
                for i in p2:
                    if not kb.is_pressed('right'):
                        print("Key released, stopping turn right action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 1500)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 1500)
                    time.sleep(d)
                time.sleep(0.5)
                for i in p2:
                    if not kb.is_pressed('right'):
                        print("Key released, stopping turn right action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i - 6, 116, 2048)
                    time.sleep(d)
                time.sleep(d2)
                self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                time.sleep(d2)
        except KeyboardInterrupt:
            print("Turn right action interrupted")
        except Exception as e:
            print(f"Error during turn right action: {e}")

    def perform_pushup_action(self):
        p2 = (7, 9, 11, 8, 10, 12)
        d = 0.01
        if self.action != 2:
            self.packet.write2ByteTxOnly(self.port, 254, 84, 600)  # P gain
            self.packet.write2ByteTxOnly(self.port, 254, 82, 0)    # I gain
            self.packet.write2ByteTxOnly(self.port, 254, 80, 15)   # D gain
            self.action = 2
            time.sleep(0.2)
        try:
            while kb.is_pressed('p'):
                for i in p2:
                    if not kb.is_pressed('p'):
                        print("Key released, stopping pushup action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 2800)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 1200)
                    time.sleep(d)
                time.sleep(2)
                for i in p2:
                    if not kb.is_pressed('p'):
                        print("Key released, stopping pushup action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 2048)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 2048)
                    time.sleep(d)
                time.sleep(2)
        except KeyboardInterrupt:
            print("Pushup action interrupted")
        except Exception as e:
            print(f"Error during pushup action: {e}")

    def perform_wave_action(self):
        p2 = (16, 17)
        p1 = (13, 14)
        d = 0.005
        if self.action != 3:
            self.packet.write2ByteTxOnly(self.port, 254, 84, 600)  # P gain
            self.packet.write2ByteTxOnly(self.port, 254, 82, 0)    # I gain
            self.packet.write2ByteTxOnly(self.port, 254, 80, 15)   # D gain
            self.action = 3
            time.sleep(0.2)
        try:
            while kb.is_pressed('h'):
                self.packet.write4ByteTxOnly(self.port, 3, 116, 2700)
                time.sleep(d)
                self.packet.write4ByteTxOnly(self.port, 6, 116, 1350)
                time.sleep(d)
                self.packet.write4ByteTxOnly(self.port, 4, 116, 2250)
                time.sleep(d)
                self.packet.write4ByteTxOnly(self.port, 5, 116, 1750)
                time.sleep(d)
                for i in p1:
                    if not kb.is_pressed('h'):
                        print("Key released, stopping wave action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 600)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i - 6, 116, 2500)
                    time.sleep(d)
                time.sleep(2)
                for _ in range(10):
                    for i in p2:
                        if not kb.is_pressed('h'):
                            print("Key released, stopping wave action")
                            self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                            return
                        self.packet.write4ByteTxOnly(self.port, i, 116, 100)
                        time.sleep(d)
                    time.sleep(0.2)
                    for i in p2:
                        if not kb.is_pressed('h'):
                            print("Key released, stopping wave action")
                            self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                            return
                        self.packet.write4ByteTxOnly(self.port, i, 116, 1000)
                        time.sleep(d)
                    time.sleep(0.2)
                time.sleep(0.2)
                self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
        except KeyboardInterrupt:
            print("Wave action interrupted")
        except Exception as e:
            print(f"Error during wave action: {e}")

    def perform_dance_action(self):
        p1 = (7, 9, 11)
        p2 = (8, 10, 12)
        d = 0.02
        d2 = 0.5
        if self.action != 4:
            self.packet.write2ByteTxOnly(self.port, 254, 84, 600)  # P gain
            self.packet.write2ByteTxOnly(self.port, 254, 82, 0)    # I gain
            self.packet.write2ByteTxOnly(self.port, 254, 80, 15)   # D gain
            self.action = 4
            time.sleep(0.2)
        try:
            while kb.is_pressed('g'):
                for i in p1:
                    if not kb.is_pressed('g'):
                        print("Key released, stopping dance action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 1500)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 1100)
                    time.sleep(d)
                time.sleep(0.5)
                for i in p2:
                    if not kb.is_pressed('g'):
                        print("Key released, stopping dance action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i - 6, 116, 1550)
                    time.sleep(d)
                time.sleep(0.5)
                for i in p2:
                    if not kb.is_pressed('g'):
                        print("Key released, stopping dance action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i - 6, 116, 2048)
                    time.sleep(d)
                time.sleep(d2)
                self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                time.sleep(d2)
                for i in p2:
                    if not kb.is_pressed('g'):
                        print("Key released, stopping dance action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 1500)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 1100)
                    time.sleep(d)
                time.sleep(0.5)
                for i in p1:
                    if not kb.is_pressed('g'):
                        print("Key released, stopping dance action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i - 6, 116, 2600)
                    time.sleep(d)
                time.sleep(d2)
                for i in p1:
                    if not kb.is_pressed('g'):
                        print("Key released, stopping dance action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i - 6, 116, 2048)
                    time.sleep(d)
                time.sleep(d2)
                self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                time.sleep(d2)
        except KeyboardInterrupt:
            print("Dance action interrupted")
        except Exception as e:
            print(f"Error during dance action: {e}")

def perform_alt_action(self):
    p1 = (7, 9, 11)
    p2 = (8, 10, 12)
    d = 0.02
    d2 = 0.5

    # Set PID values
    self.packet.write2ByteTxOnly(self.port, 254, 84, 600)  # P gain
    self.packet.write2ByteTxOnly(self.port, 254, 82, 0)    # I gain
    self.packet.write2ByteTxOnly(self.port, 254, 80, 15)   # D gain

    try:
        while True:
            if kb.is_pressed('t'):
                for i in p1:
                    if not kb.is_pressed('t'):
                        print("Key released, stopping pushup action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 1500)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 1100)
                    time.sleep(d)

                time.sleep(0.5)
                self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                time.sleep(d2)

                for i in p2:
                    if not kb.is_pressed('t'):
                        print("Key released, stopping pushup action")
                        self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                        return
                    self.packet.write4ByteTxOnly(self.port, i, 116, 1500)
                    time.sleep(d)
                    self.packet.write4ByteTxOnly(self.port, i + 6, 116, 1100)
                    time.sleep(d)

                time.sleep(0.5)
                self.packet.write4ByteTxOnly(self.port, 254, 116, 2048)
                time.sleep(d2)
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Action interrupted")
    except Exception as e:
        print(f"Error during action: {e}")



def main():
    DEVICE_NAME = 'COM9'  
    BAUDRATE = 57600   
    GAIT_FILE = "C:\\Users\\heman\\OneDrive\\Studies\\Hexy\\Motor_Control\\WALK\\Latest_Walk_inverted.npy"

    hexapod = HexapodController(DEVICE_NAME, BAUDRATE, GAIT_FILE)
    
    try:
        if hexapod.port.openPort():
            print(f"Succeeded to open port {DEVICE_NAME}")
        else:
            print("Failed to open port")
            return      
        if hexapod.port.setBaudRate(BAUDRATE):
            print(f"Succeeded to set baudrate to {BAUDRATE}")
        else:
            print("Failed to change baudrate")
            return
        hexapod.packet.write4ByteTxOnly(hexapod.port, 254, 116, 2048)
        time.sleep(1)

        while True:
            if kb.is_pressed('up'):
                hexapod.move_forward()
            elif kb.is_pressed('down'):
                hexapod.move_backward()
            elif kb.is_pressed('left'):
                hexapod.rotate_left()
            elif kb.is_pressed('right'):
                hexapod.rotate_right()
            elif kb.is_pressed('p'):
                hexapod.perform_pushup_action()
            elif kb.is_pressed('h'):
                hexapod.perform_wave_action()
            elif kb.is_pressed('g'):
                hexapod.perform_dance_action()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nProgram stopped")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        if hexapod.port.is_open:
            hexapod.port.closePort()
            print("Port closed")

if __name__ == "__main__":
    main()