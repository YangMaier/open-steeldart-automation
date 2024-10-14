from skimage.color import label2rgb
from skimage.segmentation import slic

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
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
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

    cv.namedWindow('sliders')
    cv.createTrackbar('blur', 'sliders', 0, 100, nothing)
    cv.createTrackbar('kmeans k', 'sliders', 3, 1000, nothing)
    cv.createTrackbar('compactness', 'sliders', 3, 20, nothing)

    cv.setTrackbarPos('blur', 'sliders', 5)
    cv.setTrackbarPos('kmeans k', 'sliders', 300)
    cv.setTrackbarPos('compactness', 'sliders', 5)

    while rval:
        rval, frame = cam.read()

        blur_val = cv.getTrackbarPos('blur', 'sliders')
        if blur_val % 2 == 0:
            blur_val += 1

        kmeans_k = cv.getTrackbarPos('kmeans k', 'sliders')
        compactness = cv.getTrackbarPos('compactness', 'sliders')

        blur = cv.GaussianBlur(frame, (blur_val, blur_val), 0)

        astronaut_segments = slic(blur, n_segments=kmeans_k, compactness=compactness)

        label_rgb = label2rgb(astronaut_segments, blur, kind='avg')

        cv.imshow("blur", blur)
        cv.imshow("frame", frame)
        cv.imshow("segments", label_rgb)

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


