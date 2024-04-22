'''''
import numpy as np
from matplotlib import pyplot as plt
import cv2

# Load calibration parameters (intrinsic matrix and distortion coefficients)
intrinsic_matrix = np.load('matrices/intrinsic_matrix.npy')
distortion_coeffs = np.load('matrices/distortion_coefficients.npy')

left_image = cv2.imread('depth_map/left_frame.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('depth_map/right_frame.png', cv2.IMREAD_GRAYSCALE)

# Undistort left and right images
undistorted_left_image = cv2.undistort(left_image, intrinsic_matrix, distortion_coeffs)
undistorted_right_image = cv2.undistort(right_image, intrinsic_matrix, distortion_coeffs)


stereo = cv2.StereoBM_create(numDisparities=64, blockSize=21)  # Adjust parameters as needed
depth = stereo.compute(left_image, right_image)

# Save the depth map
cv2.imwrite('depth_map/depth_map.png', depth)

# Display left and right images
cv2.imshow("Left", left_image)
cv2.imshow("Right", right_image)


# Display depth map
plt.imshow(depth, cmap='jet')  # Use a suitable colormap
plt.axis('off')
plt.colorbar()
plt.show()
'''''

import numpy as np
from matplotlib import pyplot as plt
import cv2

# Load calibration parameters (intrinsic matrix and distortion coefficients)
intrinsic_matrix = np.load('matrices/intrinsic_matrix.npy')
distortion_coeffs = np.load('matrices/distortion_coefficients.npy')

# Load left and right images
left_image = cv2.imread('depth_map/left_frame.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('depth_map/right_frame.png', cv2.IMREAD_GRAYSCALE)

# Undistort left and right images
undistorted_left_image = cv2.undistort(left_image, intrinsic_matrix, distortion_coeffs)
undistorted_right_image = cv2.undistort(right_image, intrinsic_matrix, distortion_coeffs)

# Create a StereoBM object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)  # Adjust parameters as needed

# Compute the depth map using undistorted images
depth = stereo.compute(undistorted_left_image, undistorted_right_image)
cv2.imwrite('depth_map/depth_map.png', depth)
print(depth)

# Save the depth map
np.save('matrices/depth_map.npy', depth)

# Display left and right images
cv2.imshow("Undistorted Left", undistorted_left_image)
cv2.imshow("Undistorted Right", undistorted_right_image)

# Display depth map
plt.imshow(depth, cmap='jet')  # Use a suitable colormap
plt.axis('off')
plt.colorbar()
plt.show()

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
