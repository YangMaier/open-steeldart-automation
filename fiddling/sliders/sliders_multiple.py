import os
import pathlib

import cv2 as cv
import numpy as np

from fiddling.sliders.kmeans import get_kmeans


def nothing(x):
    pass


def canny_sliders():
    cam_id = 2
    cam = cv.VideoCapture(cam_id, cv.CAP_DSHOW)
    # cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    # cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    # cam.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    # cam.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
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

    grey_stub = np.zeros_like(frame)

    cv.namedWindow('sliders')
    cv.createTrackbar('blur', 'sliders', 0, 20, nothing)
    # cv.createTrackbar('sharpen', 'sliders', 0, 20, nothing)
    cv.createTrackbar('kmeans k', 'sliders', 1, 10, nothing)
    cv.createTrackbar('canny lower', 'sliders', 0, 255, nothing)
    cv.createTrackbar('canny upper', 'sliders', 0, 255, nothing)

    cv.setTrackbarPos('blur', 'sliders', 7)
    # cv.setTrackbarPos('sharpen', 'sliders', 9)
    cv.setTrackbarPos('kmeans k', 'sliders', 5)
    cv.setTrackbarPos('canny lower', 'sliders', 80)
    cv.setTrackbarPos('canny upper', 'sliders', 120)

    while rval:
        rval, bgr = cam.read()

        canny_lower = cv.getTrackbarPos('canny lower', 'sliders')
        canny_upper = cv.getTrackbarPos('canny upper', 'sliders')

        blur_val = cv.getTrackbarPos('blur', 'sliders')
        if blur_val % 2 == 0:
            blur_val += 1

        # sharpen_val = cv.getTrackbarPos('sharpen', 'sliders')
        # if sharpen_val % 2 == 0:
        #     sharpen_val += 1
        #
        # gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        # blur = cv.GaussianBlur(gray, (blur_val, blur_val), 0)
        # # sharpen = cv.filter2D(blur, -1, np.array([[-1, -1, -1], [-1, sharpen_val, -1], [-1, -1, -1]]))
        # canny_sharpen = cv.Canny(sharpen, canny_lower, canny_upper)

        blur_bgr = cv.GaussianBlur(bgr, (blur_val, blur_val), 0)
        # sharpen_bgr = cv.filter2D(blur_bgr, -1, np.array([[-1, -1, -1], [-1, sharpen_val, -1], [-1, -1, -1]]))

        kmeans_k_val = cv.getTrackbarPos('kmeans k', 'sliders')
        segmented_image = get_kmeans(blur_bgr, kmeans_k_val)
        # cv.imshow("blur", blur)
        # cv.imshow("canny sharpen", canny_sharpen)
        # cv.imshow("canny sharpen inverted", canny_sharpen_inverted)
        # cv.imshow("sharpen", sharpen)
        # cv.imshow("sharpen_inverted", sharpen_inverted)
        # cv.imshow("scharr", scharr_edges)
        cv.imshow("sliders", grey_stub)
        cv.imshow("segmented image", segmented_image)

        key = cv.waitKey(20)
        if key == ord('c'):
            cv.imwrite('{}_{}.{}'.format(base_path, str(frame_num).zfill(digit), 'png'), bgr)
            print("Screenshot saved!")
            frame_num += 1
        if key == 27:  # exit on ESC
            cam.release()
            cv.destroyAllWindows()
            break


canny_sliders()


