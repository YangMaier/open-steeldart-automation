import os
import pathlib

import cv2 as cv
import numpy as np


def nothing(x):
    pass


def sliders():
    cam_id = 2
    cam = cv.VideoCapture(cam_id)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
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
    cv.createTrackbar('blur', 'sliders', 0, 10, nothing)
    cv.createTrackbar('ddepth', 'sliders', 0, 255, nothing)
    cv.createTrackbar('kernel_size', 'sliders', 0, 31, nothing)
    cv.createTrackbar('scale', 'sliders', 0, 20, nothing)
    cv.createTrackbar('delta', 'sliders', 0, 1000, nothing)

    cv.setTrackbarPos('blur', 'sliders', 5)
    cv.setTrackbarPos('ddepth', 'sliders', 1)
    cv.setTrackbarPos('kernel_size', 'sliders', 3)
    cv.setTrackbarPos('scale', 'sliders', 18)
    cv.setTrackbarPos('delta', 'sliders', 255)

    while rval:
        rval, frame = cam.read()

        ycrcb_img = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(ycrcb_img)
        y_equalized = cv.equalizeHist(y)
        ycrcb = cv.merge((y_equalized, cr, cb))
        equalized_bgr = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

        ddepth = cv.getTrackbarPos('ddepth', 'sliders')
        kernel_size = cv.getTrackbarPos('kernel_size', 'sliders')
        if kernel_size % 2 == 0:
            kernel_size += 1
        scale = cv.getTrackbarPos('scale', 'sliders')
        delta = cv.getTrackbarPos('delta', 'sliders')
        blur_val = cv.getTrackbarPos('blur', 'sliders')
        if blur_val % 2 == 0:
            blur_val += 1

        gray = cv.cvtColor(equalized_bgr, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (blur_val, blur_val), 0)
        dst = 0
        laplacian = cv.Laplacian(blur, dst, ddepth, kernel_size, scale, delta)
        cv.imshow("laplacian", laplacian)
        laplacian_inverted = cv.bitwise_not(laplacian)
        cv.imshow("laplacian_inverted", laplacian_inverted)

        thresh_laplace_inv = cv.threshold(laplacian_inverted, 254, 255, cv.THRESH_OTSU)[1]
        cv.imshow("thresh_laplace", thresh_laplace_inv)
        contours, _ = cv.findContours(thresh_laplace_inv, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours_filtered = [cnt for cnt in contours if cv.contourArea(cnt) > 40]
        laplace_inv_thresh_contours = np.zeros(thresh_laplace_inv.shape)
        cv.drawContours(laplace_inv_thresh_contours, contours_filtered, -1, (255, 255, 255), -1)
        cv.imshow("laplace_inv_thresh_contours", laplace_inv_thresh_contours)
        # contours_eroded = cv.erode(laplace_inv_thresh_contours, np.ones((3, 3), np.uint8))
        # cv.imshow("contours_eroded", contours_eroded)
        # laplacian = cv.Laplacian(blur, cv.CV_64F)

        # cv.imshow("frame", frame)

        key = cv.waitKey(20)
        if key == ord('c'):
            cv.imwrite('{}_{}.{}'.format(base_path, str(frame_num).zfill(digit), 'png'), frame)
            print("Screenshot saved!")
            frame_num += 1
        if key == 27:  # exit on ESC
            cam.release()
            cv.destroyAllWindows()
            break


sliders()


