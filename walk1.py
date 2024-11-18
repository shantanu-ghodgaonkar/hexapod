import os, sys, ctypes, time
import dynamixel_sdk as dxl 
import numpy as np

def walk1(port, packet, gait_file, steps=5, delay=0.02):
    # time.sleep(1)
    try:
        gait_angles = np.load(gait_file, allow_pickle=True)
        print(f"Loaded gait file with {gait_angles.shape[0]} frames")
        # print(np.max(gait_angles))

    except Exception as e:
        print(f"Error loading gait file: {e}")
        return
    
    group_sync_write = dxl.GroupSyncWrite(port, packet, 116, 4)
    
    try:
        packet.write1ByteTxOnly(port, 254, 65, 1)  
        packet.write1ByteTxOnly(port, 254, 64, 1)  
        time.sleep(1)
        
        print(f"Starting walking")
        for step in range(steps):

            print(f"Step {step + 1}/{steps}")
            seq=(1,7,13,2,8,14,3,9,15,4,10,16,5,11,17,6,12,18)
            for frame in range(gait_angles.shape[0]):
                time.sleep(0.1)
                group_sync_write.clearParam()
                
                for motor_id in range(0,18,1):
                    # angle = int((gait_angles[frame, motor_id - 1] * 2048 / np.pi) + 2048)
                    angle = int(gait_angles[frame,motor_id])
                    angle = max(0, min(4095, angle))
                    print(f"Motor ID {seq[motor_id]} Angle {str((angle-2048)*180/2048)} deg")
                
                    param = [
                        dxl.DXL_LOBYTE(dxl.DXL_LOWORD(angle)),
                        dxl.DXL_HIBYTE(dxl.DXL_LOWORD(angle)),
                        dxl.DXL_LOBYTE(dxl.DXL_HIWORD(angle)),
                        dxl.DXL_HIBYTE(dxl.DXL_HIWORD(angle))
                    ]
                    group_sync_write.addParam(seq[motor_id], param)
                print(f"Frame Complete {frame}")
                result = group_sync_write.txPacket()
                # if result != dxl.COMM_SUCCESS:
                    # print(f"Failed to send packet: {packet.getTxRxResult(result)}")
            time.sleep(delay)
            packet.write4ByteTxOnly(port,254,116,2048)
            time.sleep(delay)    
                # time.sleep(delay)
            
        print("Walking completed")
        
    except Exception as e:
        print(f"Error during walking: {e}")
        
    finally:
        group_sync_write.clearParam()
        packet.write1ByteTxOnly(port, 254, 65, 0)
        # packet.write1ByteTxOnly(port, 254, 64, 0)
        print("Motors disabled")

def main():
    DEVICENAME = 'COM7'  
    BAUDRATE = 57600   
    GAIT_FILE = "C:\\Users\\heman\\OneDrive\\Studies\\Hexy\\Motor_Control\\WALK\\gait_angles_1_converted_1.npy"  
    port = dxl.PortHandler(DEVICENAME)
    packet = dxl.PacketHandler(2.0)
    
    try:
        if port.openPort():
            print(f"Succeeded to open port {DEVICENAME}")
        else:
            print("Failed to open port")
            return      
        if port.setBaudRate(BAUDRATE):
            print(f"Succeeded to set baudrate to {BAUDRATE}")
        else:
            print("Failed to change baudrate")
            return
        packet.write4ByteTxOnly(port,254,116,2048)
        time.sleep(1)

        walk1(port, packet, GAIT_FILE)
        time.sleep(1)
        # packet.write4ByteTxOnly(port,254,116,2048)
        time.sleep(1)
    except KeyboardInterrupt:
        print("\nProgram stopped")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        if port.is_open:
            port.closePort()
            print("Port closed")

if __name__ == "__main__":
    main()