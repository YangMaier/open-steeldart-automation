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

    cv.namedWindow('sobel')
    cv.createTrackbar('blur', 'sobel', 0, 20, nothing)
    # cv.createTrackbar('ddepth', 'sobel', 0, 255, nothing)
    cv.createTrackbar('kernel_size', 'sobel', 0, 31, nothing)
    cv.createTrackbar('scale', 'sobel', 0, 255, nothing)
    cv.createTrackbar('delta', 'sobel', 0, 255, nothing)
    cv.createTrackbar('erode_kernel', 'sobel', 0, 20, nothing)

    cv.setTrackbarPos('blur', 'sobel', 1)
    cv.setTrackbarPos('kernel_size', 'sobel', 1)
    cv.setTrackbarPos('scale', 'sobel', 1)
    cv.setTrackbarPos('delta', 'sobel', 0)

    while rval:
        rval, frame = cam.read()

        # ddepth = cv.getTrackbarPos('ddepth', 'sobel')
        kernel_size = cv.getTrackbarPos('kernel_size', 'sobel')
        if kernel_size % 2 == 0:
            kernel_size += 1
        scale = cv.getTrackbarPos('scale', 'sobel')
        delta = cv.getTrackbarPos('delta', 'sobel')
        blur_val = cv.getTrackbarPos('blur', 'sobel')
        if blur_val % 2 == 0:
            blur_val += 1

        erode_kernel = cv.getTrackbarPos('erode_kernel', 'sobel')
        if erode_kernel % 2 == 0:
            erode_kernel += 1

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (blur_val, blur_val), 0)
        # sobel = cv.sobel(blur, cv.CV_64F)
        grad_x = cv.Sobel(blur, cv.CV_16S, 1, 0, ksize=kernel_size, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

        # Gradient-Y
        # grad_y = cv.Scharr(gray,ddepth,0,1)
        grad_y = cv.Sobel(blur, cv.CV_16S, 0, 1, ksize=kernel_size, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)

        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        erode = cv.erode(grad, np.ones((erode_kernel, erode_kernel), np.uint8))

        cv.imshow("sobel", grad)
        cv.imshow("blur", blur)
        cv.imshow("frame", frame)
        cv.imshow("erode", erode)

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


