import numpy as np
import cv2
import glob
import math

# Define the checkerboard dimensions
checkerboard_size = (7, 7)  # Corners (width-1, height-1)
square_size = 49.21  # Set the actual size of the squares on your checkerboard in mm

# Prepare object points (like (0,0,0), (1,0,0), (2,0,0) ..., assuming the chessboard is flat on the z=0 plane)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# List of paths to calibration images
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    # print(f"Processing {fname}: Corners found = {ret}")

    # If found, refine corner locations and add them to the arrays
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        #  Print refined corners and corresponding world coordinates

        # Draw each corner on the image
        for corner in corners2:
            cv2.circle(img, tuple(corner.ravel().astype(int)), 10, (0, 255, 0), -1)

        # Optionally, display the image with drawn corners
        cv2.imshow('Corners', img)
        cv2.waitKey(500)  # Wait for 500ms

        # Optionally, save the image with corners drawn
        cv2.imwrite(f'points_drawn/corners_drawn_{fname.split("/")[-1]}', img)
        print(f"Refined corners for {fname}: {corners2.reshape(-1, 2)}")
        print(f"World coordinates (object points) for {fname}: {objp}")
    else:
        print(f"No corners found in {fname}.")

# Perform camera calibration to obtain intrinsic matrix, distortion coefficients, and extrinsic parameters for each image
if len(objpoints) > 0 and len(imgpoints) > 0:  # Check if we have enough data for calibration
    ret, mtx, dist, rotation_vecs, translation_vecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(": \n", ret)
    print("Intrinsic Matrix: \n", mtx)
    print("Rotational Matrix: \n", rotation_vecs)
    print("Translational Matrix: \n", translation_vecs)
    print("Camera matrix: \n", mtx)
    print("Distortion coefficients: \n", dist)
    np.save('matrices/intrinsic_matrix.npy', mtx)
    np.save('matrices/distortion_coefficients.npy', dist)


    # Select the specific image's extrinsic parameters (using Image 8 as an example)
    selected_index = 7  # Assuming you have at least 9 images and want the 9th one
    if selected_index < len(rotation_vecs):  # Check if the selected index is valid
        rotation_vec = rotation_vecs[selected_index]
        translation_vec = translation_vecs[selected_index]
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rotation_vec)
        extrinsic_matrix = np.hstack((R, translation_vec))
        np.save('rotation_vectors.npy', rotation_vecs)
        np.save('translation_vectors.npy', translation_vecs)
        # Print results
        print(f"Rotation matrix for image {selected_index}:", R)
        print(f"Translation matrix for image {selected_index}:", translation_vec)
        print(f"Extrinsic matrix for image {selected_index}: \n", extrinsic_matrix)
        # If you have a method to compute Euler angles from R, you can print them here
        # print("Euler angles:", euler_angles)  # Assuming you have a method to convert
        # Compute the Euler angles
        yaw = math.atan2(R[1, 0], R[0, 0])  # atan2(r21, r11)
        pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1]**2 + R[2, 2]**2))  # atan2(-r31, sqrt(r32^2 + r33^2))
        roll = math.atan2(R[2, 1], R[2, 2])  # atan2(r32, r33)

        # Convert the angles from radians to degrees
        yaw_deg = np.degrees(yaw)
        pitch_deg = np.degrees(pitch)
        roll_deg = np.degrees(roll)

        # Print the results
        print(f"Yaw (ψ): {yaw_deg} degrees")
        print(f"Pitch (θ): {pitch_deg} degrees")
        print(f"Roll (φ): {roll_deg} degrees")
    else:
        print(f"Selected index {selected_index} is out of range.")
else:
    print("Not enough data for calibration.")
