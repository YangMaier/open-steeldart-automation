import os
import pathlib

import cv2 as cv
import numpy as np


def nothing(x):
    pass


def canny_sliders():
    cam_id = 2
    cam = cv.VideoCapture(cam_id, cv.CAP_DSHOW)
    # cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    # cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_path = pathlib.Path().absolute().joinpath("../media/fiddling")
    os.makedirs(frame_path, exist_ok=True)
    frame_basename = 'sample_video_cap'
    base_path = os.path.join(frame_path, frame_basename)
    frame_num = 0

    digit = len(str(int(cam.get(cv.CAP_PROP_FRAME_COUNT))))

    # cv.namedWindow(preview_name)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    # cv.namedWindow('laplacian')
    # cv.createTrackbar('blur', 'laplacian', 0, 10, nothing)
    # cv.createTrackbar('ddepth', 'laplacian', 0, 255, nothing)
    # cv.createTrackbar('kernel_size', 'laplacian', 0, 10, nothing)
    # cv.createTrackbar('scale', 'laplacian', 0, 255, nothing)
    # cv.createTrackbar('delta', 'laplacian', 0, 255, nothing)
    # cv.setTrackbarPos('blur', 'laplacian', 3)
    # cv.setTrackbarPos('ddepth', 'laplacian', 1)
    # cv.setTrackbarPos('kernel_size', 'laplacian', 3)
    # cv.setTrackbarPos('scale', 'laplacian', 1)
    # cv.setTrackbarPos('delta', 'laplacian', 0)

    while rval:
        rval, frame = cam.read()

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        r, g, b = cv.split(rgb)

        cv.imshow('red_transparent', r)
        cv.imshow('green', g)
        cv.imshow('blue', b)
        #
        # ddepth = cv.getTrackbarPos('ddepth', 'laplacian')
        # kernel_size = cv.getTrackbarPos('kernel_size', 'laplacian')
        # scale = cv.getTrackbarPos('scale', 'laplacian')
        # delta = cv.getTrackbarPos('delta', 'laplacian')
        # blur_val = cv.getTrackbarPos('blur', 'laplacian')
        #
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # blur = cv.GaussianBlur(gray, (blur_val, blur_val), 0)
        # dst = 0
        # laplacian = cv.Laplacian(blur, dst, ddepth, kernel_size, scale, delta)
        # laplacian = cv.Laplacian(blur, cv.CV_64F)
        #
        # cv.imshow("laplacian", laplacian)
        cv.imshow("frame", frame)

        key = cv.waitKey(20)
        if key == ord('c'):
            cv.imwrite('{}_{}.{}'.format(base_path, str(frame_num).zfill(digit), 'png'), frame)
            print("Screenshot saved!")
            frame_num += 1
        if key == 27:  # exit on ESC
            cam.release()
            cv.destroyAllWindows()
            break


canny_sliders()


