import pyzed.sl as sl
import numpy as np

# Initialize the ZED camera
zed = sl.Camera()

# Set initialization parameters
init_params = sl.InitParameters()
init_params.coordinate_units = sl.UNIT.METER  # Set the unit to meters
init_params.camera_resolution = sl.RESOLUTION.HD720  # You can change this if needed

# Open the ZED camera
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

# Enable depth mode for point cloud
runtime_parameters = sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD or FILL

# Create a Mat object to store the point cloud
point_cloud = sl.Mat()

try:
    while True:
        # Capture a new image and compute the depth map
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # Get the width and height of the point cloud
            width = point_cloud.get_width()
            height = point_cloud.get_height()

            # Access the point cloud data
            # Create a NumPy array of shape (height, width, 4) to hold (X, Y, Z, RGBA)
            point_cloud_np = point_cloud.get_data().reshape(height, width, 4)

            # Extract the 3D coordinates (X, Y, Z)
            xyz_points = point_cloud_np[:, :, :3]

            # Print out some point cloud data (e.g., point at the center of the image)
            center_x, center_y = width // 2, height // 2
            center_point = xyz_points[center_y, center_x]
            print(f"3D coordinates of center point: X = {center_point[0]:.3f}, Y = {center_point[1]:.3f}, Z = {center_point[2]:.3f}")

            # Optional: Perform operations with the point cloud data (e.g., saving or processing)

except KeyboardInterrupt:
    # Close the camera when the script is interrupted
    zed.close()
    print("Camera closed")

