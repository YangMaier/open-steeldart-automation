import os
import pathlib

import cv2 as cv
import numpy as np


def nothing(x):
    pass


def corner_harris_sliders():
    cam_id = 0
    cam = cv.VideoCapture(cam_id)
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

    cv.namedWindow('cornerharris')
    cv.createTrackbar('blur', 'cornerharris', 0, 20, nothing)
    cv.createTrackbar('sharpen', 'cornerharris', 0, 20, nothing)
    cv.createTrackbar('blockSize', 'cornerharris', 2, 20, nothing)
    cv.createTrackbar('ksize', 'cornerharris', 1, 20, nothing)
    cv.createTrackbar('k', 'cornerharris', 0, 200, nothing)

    cv.setTrackbarPos('blur', 'cornerharris', 5)
    cv.setTrackbarPos('sharpen', 'cornerharris', 5)
    cv.setTrackbarPos('blockSize', 'cornerharris', 3)
    cv.setTrackbarPos('ksize', 'cornerharris', 13)
    cv.setTrackbarPos('k', 'cornerharris', 1)

    while rval:
        rval, frame = cam.read()

        blur_val = cv.getTrackbarPos('blur', 'cornerharris')
        if blur_val % 2 == 0:
            blur_val += 1

        sharpen_val = cv.getTrackbarPos('sharpen', 'cornerharris')
        if sharpen_val % 2 == 0:
            sharpen_val += 1

        blockSize = cv.getTrackbarPos('blockSize', 'cornerharris')
        if blockSize % 2 == 0:
            blockSize += 1
        ksize = cv.getTrackbarPos('ksize', 'cornerharris')
        if ksize % 2 == 0:
            ksize += 1
        k = cv.getTrackbarPos('k', 'cornerharris')
        k = k * 0.001

        blur = cv.bilateralFilter(frame, 20, 30, 30)

        gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        # blur = cv.GaussianBlur(gray, (blur_val, blur_val), 0)
        # sharpen = cv.filter2D(blur, -1, np.array([[-1, -1, -1], [-1, sharpen_val, -1], [-1, -1, -1]]))
        corners = cv.cornerHarris(gray, blockSize, ksize, k)

        cv.imshow("cornerharris", corners)
        cv.imshow("blur", blur)
        # cv.imshow("sharpen", sharpen)
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


corner_harris_sliders()


