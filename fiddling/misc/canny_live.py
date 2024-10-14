import os
import pathlib

import cv2 as cv
import numpy as np


def canny_live():
    cam_id = 0
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

    while rval:
        rval, frame = cam.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        canny = cv.Canny(blur, 100, 200)

        x_sobel = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=7)
        y_sobel = cv.Sobel(blur, cv.CV_64F, 1, 0, ksize=7)
        xy_sobel = np.hstack((x_sobel, y_sobel))
        laplacian = cv.Laplacian(blur, cv.CV_64F)

        cv.imshow("edges", canny)

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


canny_live()
