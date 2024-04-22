import cv2
import numpy as np

# Load an image
image = cv2.imread('frames/frame_0004.jpg', cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# Compute the integral image
integral_image = cv2.integral(image)

# The result is a numpy array where the integral image is shifted by one pixel to the right and down
print(integral_image)


def compute_integral_image(image):
    # Get the dimensions of the image
    height, width = image.shape

    # Create an array to store the integral image with an extra row and column (filled with zeros)
    integral_image = np.zeros((height + 1, width + 1), dtype=np.uint32)

    # Compute the integral image
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            # Sum values from the top to the current pixel, left to the current pixel, and subtract the top-left overlap
            integral_image[y, x] = (image[y - 1, x - 1] +
                                    integral_image[y - 1, x] +
                                    integral_image[y, x - 1] -
                                    integral_image[y - 1, x - 1])

    return integral_image[1:, 1:]  # Remove the first row and column to shift back

print('##################################################################')
# Example of loading an image and computing its integral image manually
image = cv2.imread('frames/frame_0004.jpg', cv2.IMREAD_GRAYSCALE)
integral_img = compute_integral_image(image)
print(integral_img)
