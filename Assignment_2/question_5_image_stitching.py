import cv2
import numpy as np
import depthai as dai


# Function to stitch two images together
def image_stitch(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()

    # Compute descriptors for the first image
    kp1, des1 = orb.detectAndCompute(gray1, None)
    if des1 is None:
        print("No descriptors computed for the first image")
        return None

    # Compute descriptors for the second image
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des2 is None:
        print("No descriptors computed for the second image")
        return None

    # Convert descriptors to np.float32
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    # Match descriptors using FLANN matcher
    flann = cv2.FlannBasedMatcher_create()
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches using ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Estimate homography if enough good matches are found
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp img1 to img2 using estimated homography
        warped_img = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))

        # Blend the warped image and img2
        warped_img[:, 0:img2.shape[1]] = img2

        return warped_img
    else:
        print("Not enough good matches found")
        return None


# Initialize DepthAI pipeline
pipeline = dai.Pipeline()

# Configure Oak-D Lite camera nodes
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(416, 416)  # Set desired preview size
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")

cam_rgb.preview.link(xout_rgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queue for RGB frames
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    # Capture the first frame
    frame = q_rgb.get().getCvFrame()

    # Use the first frame as the previous frame for stitching
    prev_frame = frame.copy()

    while True:
        # Capture frame-by-frame
        frame = q_rgb.get().getCvFrame()

        # Stitch the current frame with the previous frame
        stitched_img = image_stitch(prev_frame, frame)

        if stitched_img is not None:
            # Display the stitched image
            cv2.imshow('Stitched Image', stitched_img)

                # Save the stitched image to disk
            cv2.imwrite('stitched_image.png', stitched_img)

        # Update the previous frame
        prev_frame = frame.copy()

        # Wait for 100 milliseconds
        cv2.waitKey(300)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close all windows
cv2.destroyAllWindows()
