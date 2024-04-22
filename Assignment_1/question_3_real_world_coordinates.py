import cv2
import numpy as np

# Load left and right images
left_image = cv2.imread('depth_map/left_frame.png')
right_image = cv2.imread('depth_map/right_frame.png')

# Get dimensions of left image
height_left, width_left, channels_left = left_image.shape
print("Dimensions of left image:", width_left, "x", height_left)

# Get dimensions of right image
height_right, width_right, channels_right = right_image.shape
print("Dimensions of right image:", width_right, "x", height_right)

# Load calibration parameters (intrinsic matrix and distortion coefficients)
intrinsic_matrix = np.load('matrices/intrinsic_matrix.npy')
distortion_coeffs = np.load('matrices/distortion_coefficients.npy')


'''''
# Load depth map
depth_map = cv2.imread('depth_map/depth_map.png', cv2.IMREAD_UNCHANGED)


# Define intervals for sampling points
x_interval = 50
y_interval = 50

# Loop through the depth map and sample points at defined intervals
for y in range(0, depth_map.shape[0], y_interval):
    for x in range(0, depth_map.shape[1], x_interval):
        # Get depth value from depth map at corresponding pixel location
        depth = depth_map[y, x]

        # Convert pixel coordinates to homogeneous coordinates
        pixel_coordinates = np.array([[x], [y], [1]])

        # Apply inverse calibration transformation to get normalized coordinates
        normalized_coordinates = np.dot(np.linalg.inv(intrinsic_matrix), pixel_coordinates)
        normalized_coordinates *= depth  # Scale by depth to get real-world coordinates

        print(f"Pixel coordinates: ({x}, {y}) --> Normalized coordinates: ({normalized_coordinates[0]}, {normalized_coordinates[1]}, {normalized_coordinates[2]})")
'''''

# Load depth map
depth_map = np.load('matrices/depth_map.npy')

# Define intervals for sampling points
x_interval = 50
y_interval = 50

# Loop through the depth map and sample points at defined intervals
for y in range(0, depth_map.shape[0], y_interval):
    for x in range(0, depth_map.shape[1], x_interval):
        # Get depth value from depth map at corresponding pixel location
        depth = depth_map[y, x]

        # Convert pixel coordinates to homogeneous coordinates
        pixel_coordinates = np.array([[x], [y], [1]])

        # Apply inverse calibration transformation to get normalized coordinates
        normalized_coordinates = np.dot(np.linalg.inv(intrinsic_matrix), pixel_coordinates)
        normalized_coordinates *= depth  # Scale by depth to get real-world coordinates

        print(f"Pixel coordinates: ({x}, {y}) --> Normalized coordinates: ({normalized_coordinates[0]}, {normalized_coordinates[1]}, {normalized_coordinates[2]})")
