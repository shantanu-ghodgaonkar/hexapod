import pyzed.sl as sl
import matplotlib.pyplot as plt
import time
from collections import deque

# Initialize the ZED camera
zed = sl.Camera()

# Set initialization parameters
init_params = sl.InitParameters()
init_params.coordinate_units = sl.UNIT.METER
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

# Open the ZED camera
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

# Enable positional tracking (IMU data is part of this)
tracking_params = sl.PositionalTrackingParameters()
zed.enable_positional_tracking(tracking_params)

# Create a SensorsData object
sensors_data = sl.SensorsData()

# Lists to hold time and IMU data for plotting
times = deque(maxlen=100)  # Limit to the last 100 data points
acceleration_x = deque(maxlen=100)
acceleration_y = deque(maxlen=100)
acceleration_z = deque(maxlen=100)
angular_velocity_x = deque(maxlen=100)
angular_velocity_y = deque(maxlen=100)
angular_velocity_z = deque(maxlen=100)

# Set up the plot
plt.ion()  # Enable interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Titles and labels
ax1.set_title("Acceleration over Time")
ax1.set_ylabel("Acceleration (m/s^2)")
ax2.set_title("Angular Velocity over Time")
ax2.set_ylabel("Angular Velocity (rad/s)")
ax2.set_xlabel("Time (s)")

# Plotting loop
start_time = time.time()

try:
    while True:
        # Get the current time
        current_time = time.time() - start_time

        # Retrieve IMU data
        if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
            imu_data = sensors_data.get_imu_data()

            # Get acceleration and angular velocity
            acceleration = imu_data.get_linear_acceleration()
            angular_velocity = imu_data.get_angular_velocity()

            # Append data to the lists
            times.append(current_time)
            acceleration_x.append(acceleration[0])
            acceleration_y.append(acceleration[1])
            acceleration_z.append(acceleration[2])
            angular_velocity_x.append(angular_velocity[0])
            angular_velocity_y.append(angular_velocity[1])
            angular_velocity_z.append(angular_velocity[2])

            # Clear previous plots
            ax1.cla()
            ax2.cla()

            # Plot acceleration data
            ax1.plot(times, acceleration_x, label="Acc X", color="r")
            ax1.plot(times, acceleration_y, label="Acc Y", color="g")
            ax1.plot(times, acceleration_z, label="Acc Z", color="b")
            ax1.legend(loc="upper left")
            ax1.set_title("Acceleration over Time")
            ax1.set_ylabel("Acceleration (m/s^2)")

            # Plot angular velocity data
            ax2.plot(times, angular_velocity_x, label="Ang Vel X", color="r")
            ax2.plot(times, angular_velocity_y, label="Ang Vel Y", color="g")
            ax2.plot(times, angular_velocity_z, label="Ang Vel Z", color="b")
            ax2.legend(loc="upper left")
            ax2.set_title("Angular Velocity over Time")
            ax2.set_ylabel("Angular Velocity (rad/s)")
            ax2.set_xlabel("Time (s)")

            # Update the plots
            plt.draw()
            plt.pause(0.01)  # Small pause to allow the plot to update

except KeyboardInterrupt:
    # Close the camera and end plotting
    zed.close()
    plt.ioff()
    plt.show()
    print("Camera closed and plot finished.")
