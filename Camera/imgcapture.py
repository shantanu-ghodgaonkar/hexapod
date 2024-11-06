import pyzed.sl as sl
import numpy as np
import cv2

# Initialize the ZED camera
zed = sl.Camera()

# Set initialization parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA or QUALITY for depth
init_params.coordinate_units = sl.UNIT.METER  # Depth will be in meters

# Open the camera
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

# Create Mat objects to hold images and depth data
image = sl.Mat()
depth = sl.Mat()

# Runtime parameters (without sensing mode)
runtime_params = sl.RuntimeParameters()

# Number of images to capture
num_images = 5
image_count = 0

print("Press 'c' to capture an image, or 'q' to quit.")

try:
    while True:
        # Grab the current frame from the camera
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Show a live feed of the camera (optional)
            zed.retrieve_image(image, sl.VIEW.LEFT)
            rgb_image = image.get_data()[:, :, :3]  # Get the RGB image
            cv2.imshow("Camera Feed - Press 'c' to capture", rgb_image)

            # Wait for keyboard input
            key = cv2.waitKey(1)

            if key == ord('c'):  # 'c' to capture the image
                print(f"Capturing image {image_count+1}/{num_images}...")

                # Retrieve the depth map
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                # Convert the image and depth data to NumPy arrays
                image_np = image.get_data()[:, :, :3]  # Extract RGB channels
                depth_np = depth.get_data()  # Depth data

                # Save the image and depth data into a combined matrix
                combined_data = np.dstack((image_np, depth_np))  # Combine RGB and depth

                # Save the combined data with a unique filename
                np.save(f"image_with_depth_{image_count}.npy", combined_data)
                cv2.imwrite(f"rgb_image_{image_count}.png", image_np)  # Save RGB image

                # Normalize and save the depth image for visualization
                depth_normalized = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
                depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(f"depth_image_{image_count}.png", depth_colormap)  # Save depth image

                print(f"Image {image_count+1} and depth saved!")
                image_count += 1

                # Stop after capturing the desired number of images
                if image_count >= num_images:
                    print("Captured all images.")
                    break

            elif key == ord('q'):  # 'q' to quit the program
                print("Exiting...")
                break

except KeyboardInterrupt:
    print("Program interrupted")

# Close the camera and destroy windows
zed.close()
cv2.destroyAllWindows()
