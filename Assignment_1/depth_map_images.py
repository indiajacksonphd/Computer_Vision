'''''
import cv2
import depthai as dai
import os
import numpy as np


def get_frame(queue):
    frame = queue.get()  # get frame from queue
    return frame.getCvFrame()  # convert frame to OpenCV format and return


def get_mono_camera(pipeline, is_left):
    mono = pipeline.createMonoCamera()  # configure mono camera
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # set camera resolution

    if is_left:
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # get left camera
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # get right camera

    return mono


if __name__ == '__main__':
    pipeline = dai.Pipeline()  # define a pipeline

    mono_left = get_mono_camera(pipeline, is_left=True)  # set up left camera
    mono_right = get_mono_camera(pipeline, is_left=False)  # set up right camera

    x_out_left = pipeline.createXLinkOut()  # set output xlink for left camera
    x_out_left.setStreamName("left")  # set name xlink for left camera

    x_out_right = pipeline.createXLinkOut()  # set output xlink for right camera
    x_out_right.setStreamName("right")  # set name xlink for right camera

    mono_left.out.link(x_out_left.input)  # attach left camera to left xlink
    mono_right.out.link(x_out_right.input)  # attach right camera to right xlink

    with dai.Device(pipeline) as device:

        left_queue = device.getOutputQueue(name="left", maxSize=1)  # get output queue for left camera
        right_queue = device.getOutputQueue(name="right", maxSize=1)  # get output queue for right camera

        while True:
            left_frame = get_frame(left_queue)  # get left frame
            right_frame = get_frame(right_queue)  # get right frame

            # Display stereo pair
            cv2.imshow("Stereo Pair", np.hstack((left_frame, right_frame)))

            key = cv2.waitKey(1)  # check for keyboard input
            if key == ord('q'):
                break  # quit when 'q' is pressed
            elif key == ord('s'):
                # Save left and right images
                cv2.imwrite("depth_map/left_frame.png", left_frame)
                cv2.imwrite("depth_map/right_frame.png", right_frame)
                print("Images saved!")  # Print message to console
'''''


import cv2
import depthai as dai
import numpy as np

def get_frame(queue):
    frame = queue.get()  # get frame from queue
    return frame.getCvFrame()  # convert frame to OpenCV format and return

def get_mono_camera(pipeline, is_left):
    mono = pipeline.createMonoCamera()  # configure mono camera
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)  # set camera resolution

    if is_left:
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # get left camera
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # get right camera

    return mono

if __name__ == '__main__':
    pipeline = dai.Pipeline()  # define a pipeline

    # Set up left and right cameras
    mono_left = get_mono_camera(pipeline, is_left=True)
    mono_right = get_mono_camera(pipeline, is_left=False)

    # Set up RGB camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setInterleaved(False)

    # Create XLink outputs for left, right, and RGB cameras
    x_out_left = pipeline.createXLinkOut()
    x_out_left.setStreamName("left")

    x_out_right = pipeline.createXLinkOut()
    x_out_right.setStreamName("right")

    x_out_rgb = pipeline.createXLinkOut()
    x_out_rgb.setStreamName("rgb")

    # Link cameras to XLink outputs
    mono_left.out.link(x_out_left.input)
    mono_right.out.link(x_out_right.input)
    cam_rgb.preview.link(x_out_rgb.input)

    # Connect to the device and start the pipeline
    with dai.Device(pipeline) as device:
        left_queue = device.getOutputQueue(name="left", maxSize=1)
        right_queue = device.getOutputQueue(name="right", maxSize=1)
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1)

        while True:
            # Get frames from left, right, and RGB cameras
            left_frame = get_frame(left_queue)
            right_frame = get_frame(right_queue)
            rgb_frame = get_frame(rgb_queue)

            # Display stereo pair and RGB image
            cv2.imshow("Stereo Pair", np.hstack((left_frame, right_frame)))
            cv2.imshow("RGB Image", rgb_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break  # quit when 'q' is pressed
            elif key == ord('s'):
                # Save left, right, and RGB images
                cv2.imwrite("depth_map/left_frame.png", left_frame)
                cv2.imwrite("depth_map/right_frame.png", right_frame)
                cv2.imwrite("depth_map/rgb_frame.png", rgb_frame)
                print("Images saved!")  # Print message to console
