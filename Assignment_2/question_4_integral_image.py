import cv2
import numpy as np

# Ensure numpy does not truncate output when printing arrays
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def compute_integral_image(image):
    # Get the dimensions of the image
    height, width = image.shape

    # Create an array to store the integral image with an extra row and column (filled with zeros)
    integral_image = np.zeros((height + 1, width + 1), dtype=np.uint32)

    # Compute the integral image
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            integral_image[y, x] = (image[y - 1, x - 1] +
                                    integral_image[y - 1, x] +
                                    integral_image[y, x - 1] -
                                    integral_image[y - 1, x - 1])

    return integral_image[1:, 1:]  # Remove the first row and column to shift back

# Load an image
image = cv2.imread('frames/frame_0004.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Failed to load image.")
else:
    # Compute the integral image using OpenCV's built-in function
    integral_image = cv2.integral(image)
    print("Integral Image using OpenCV:")
    print(integral_image)

    print('##################################################################')

    # Compute the integral image manually
    integral_img = compute_integral_image(image)
    print("Manually Computed Integral Image:")
    print(integral_img)
