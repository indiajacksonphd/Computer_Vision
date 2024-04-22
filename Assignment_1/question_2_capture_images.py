import cv2
import depthai as dai
import time

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.createColorCamera()
xout = pipeline.createXLinkOut()

xout.setStreamName("rgb")
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Linking
camRgb.video.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)

    counter = 0
    while True:
        frame = qRgb.get()  # Get frame
        img = frame.getCvFrame()  # Convert frame to OpenCV format

        # Show the image
        cv2.imshow('Live Stream', img)

        # 's' key to save and move to next frame, 'q' to quit
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Save the image
            cv2.imwrite(f'calibration_images/image_{counter}.jpg', img)
            print(f'Saved calibration_images/image_{counter}.jpg')
            counter += 1
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
