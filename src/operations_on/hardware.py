import os

import numpy as np

import cv2 as cv


def list_connected_webcams_pylon():
    """
    WARNING: did not work as intended on Windows in some cases. Not reliable.
    Returns a list of all connected webcams and their IDs.

    Returns:
        list: A list of tuples containing the webcam ID and serial number of each connected webcam.

    Description:
        This function uses the `pypylon` library to enumerate all available cameras and retrieve their serial numbers, effectively giving you a list of connected webcams and their IDs. The function returns a list of tuples, where each tuple contains the webcam ID and serial number of a connected webcam. The webcam ID is an integer representing the order in which the cameras were connected, starting from 0. The serial number is a unique identifier for each webcam.

    Example:
        list_connected_webcams()
        # Returns: [0, 1]
    """
    # Create an empty list to store camera IDs
    camera_ids = []

    # Get all the available cameras
    pylon_tl_factory = pylon.TlFactory.GetInstance()
    devices = pylon_tl_factory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    # Print the ID of each connected camera
    for i, cam in enumerate(devices):
        camera_ids.append(i)
        print(f"Webcam ID: {i}, Serial Number: {cam.GetSerialNumber()}")

    return camera_ids


# https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
# customized because the stackoverflow solution was not reliable
def list_connected_webcam_ids():
    """
    Test the ports and returns the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6:  # if there are more than 5 non-working ports stop the fiddling.
        camera = cv.VideoCapture(dev_port)
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

        if not camera.isOpened():
            non_working_ports.append(dev_port)
            # print("Port %s is not working." % dev_port)
        else:
            frame_count = 0
            while frame_count < 30:
                frame_count += 1
                continue
            is_reading, img = camera.read()
            print("Can open Port %s." % dev_port)
            w = camera.get(3)
            h = camera.get(4)
            filename = f"test{dev_port}.jpg"
            cv.imwrite(filename, img)
            is_black_img = np.mean(img) == 0
            if os.path.exists(filename):
                can_save_img = True
                print("Can write img Port %s." % dev_port)
                # os.remove("test.jpg")
            else:
                can_save_img = False
            if is_reading and can_save_img and not is_black_img:
                print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                working_ports.append(dev_port)
            if not is_reading:
                print("Port %s for camera ( %s x %s) is present but does no reads." % (dev_port, h, w))
                available_ports.append(dev_port)
            if not can_save_img:
                print("Port %s for camera ( %s x %s) is present but cannot save images." % (dev_port, h, w))
                available_ports.append(dev_port)
            if is_black_img:
                print("Port %s for camera ( %s x %s) is present but presents a black image." % (dev_port, h, w))
                available_ports.append(dev_port)
        dev_port += 1
        camera.release()

    if len(working_ports) == 0:
        print("No camera is working.")

    return working_ports
