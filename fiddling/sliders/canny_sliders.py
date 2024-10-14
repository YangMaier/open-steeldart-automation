import os
import pathlib

import cv2 as cv
import numpy as np

from fiddling.sliders.kmeans import get_kmeans


def nothing(x):
    pass


def scharr_edge_detection(image):
    # https://blog.roboflow.com/edge-detection/
    # Convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Scharr operator to find the x and y gradients
    Gx = cv.Scharr(gray_image, cv.CV_64F, 1, 0)
    Gy = cv.Scharr(gray_image, cv.CV_64F, 0, 1)

    # Compute the gradient magnitude
    gradient_magnitude = cv.magnitude(Gx, Gy)

    return gradient_magnitude


def canny_sliders():
    cam_id = 0
    cam = cv.VideoCapture(cam_id)
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

    cv.namedWindow('canny')
    cv.createTrackbar('blur', 'canny', 0, 20, nothing)
    # cv.createTrackbar('sharpen', 'canny', 0, 20, nothing)
    cv.createTrackbar('canny lower', 'canny', 0, 255, nothing)
    cv.createTrackbar('canny upper', 'canny', 0, 255, nothing)
    # cv.createTrackbar('scale', 'canny', 1, 10, nothing)

    cv.setTrackbarPos('blur', 'canny', 5)
    # cv.setTrackbarPos('sharpen', 'canny', 11)
    cv.setTrackbarPos('canny lower', 'canny', 255)
    cv.setTrackbarPos('canny upper', 'canny', 255)
    # cv.setTrackbarPos('scale', 'canny', 1)

    while rval:
        rval, frame = cam.read()

        canny_lower = cv.getTrackbarPos('canny lower', 'canny')
        canny_upper = cv.getTrackbarPos('canny upper', 'canny')

        blur_val = cv.getTrackbarPos('blur', 'canny')
        if blur_val % 2 == 0:
            blur_val += 1

        # scale_val = cv.getTrackbarPos('scale', 'canny')

        # sharpen_val = cv.getTrackbarPos('sharpen', 'canny')
        # if sharpen_val % 2 == 0:
        #     sharpen_val += 1

        # blur = cv.blur(frame, (blur_val, blur_val))
        blur = cv.bilateralFilter(frame, 20, 30, 30)
        canny1 = cv.Canny(blur, canny_lower, canny_upper)
        cv.imshow("canny1", canny1)

        # resized_frame = cv.resize(frame, None, fx=1 / scale_val, fy=1 / scale_val, interpolation=cv.INTER_AREA)

        ycrcb_img = cv.cvtColor(blur, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(ycrcb_img)
        y_equalized = cv.equalizeHist(y)
        ycrcb = cv.merge((y_equalized, cr, cb))
        equalized_bgr = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

        hsv = cv.cvtColor(equalized_bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        s_v_sum = np.add(s, v, dtype=np.uint16)
        s_v_sum = np.clip(s_v_sum, 0, 255).astype(np.uint8)
        s_v_sum_inverted = cv.bitwise_not(s_v_sum)
        # _, thresh_s_v_sum_inv = cv.threshold(s_v_sum_inverted, 150, 255, cv.THRESH_BINARY)
        # cv.imshow("thresh s v sum inv", thresh_s_v_sum_inv)
        canny_sv = cv.Canny(s_v_sum_inverted, canny_lower, canny_upper)

        cv.imshow("canny sv", canny_sv)


        # resized_canny_v_area = cv.resize(canny_v, None, fx=scale_val, fy=scale_val, interpolation=cv.INTER_AREA)
        # resized_canny_v_linear = cv.resize(canny_v, None, fx=scale_val, fy=scale_val, interpolation=cv.INTER_LINEAR)
        # resized_canny_v_cubic = cv.resize(canny_v, None, fx=scale_val, fy=scale_val, interpolation=cv.INTER_CUBIC)
        # resized_canny_v_nearest = cv.resize(canny_v, None, fx=scale_val, fy=scale_val, interpolation=cv.INTER_NEAREST)
        # canny_v_dilate = cv.dilate(canny_v, np.ones((3, 3), np.uint8))
        cv.imshow('s_v_sum_inverted', s_v_sum_inverted)
        # cv.imshow('canny_svsum_dilate area', resized_canny_v_area)
        # cv.imshow('canny_svsum_dilate linear', resized_canny_v_linear)
        # cv.imshow('canny_svsum_dilate cubic', resized_canny_v_cubic)
        # cv.imshow('canny_svsum_dilate nearest', resized_canny_v_nearest)

        # gray = cv.cvtColor(equalized_bgr, cv.COLOR_BGR2GRAY)
        # blur = cv.GaussianBlur(gray, (blur_val, blur_val), 0)
        # sharpen = cv.filter2D(blur, -1, np.array([[-1, -1, -1], [-1, sharpen_val, -1], [-1, -1, -1]]))
        # canny = cv.Canny(blur, canny_lower, canny_upper)
        canny_dilate = cv.dilate(canny_sv, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), iterations=2)
        cv.imshow('canny_dilate', canny_dilate)
        canny_sv_dilate_inverted = cv.bitwise_not(canny_dilate)
        # canny_sv_dilate_inverted = cv.dilate(canny_sv_dilate_inverted, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=1)
        cv.imshow('canny_sv_dilate_inverted', canny_sv_dilate_inverted)
        masked_frame = cv.bitwise_and(frame, frame, mask=canny_sv_dilate_inverted)
        cv.imshow('masked frame', masked_frame)

        contours, _ = cv.findContours(canny_sv_dilate_inverted, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_img = np.zeros_like(canny_sv_dilate_inverted)
        contours_filtered = [cnt for cnt in contours if len(cnt) > 40]
        cv.drawContours(contours_img, contours_filtered, -1, (255, 255, 255), -1)
        cv.imshow('contours', contours_img)

        # canny_drawn = np.zeros_like(canny)
        # cv.drawContours(canny_drawn, contours, -1, (255, 255, 255), 2)
        # contours_filtered = [cnt for cnt in contours if len(cnt) > 10]
        # contours_filtered_img = np.zeros_like(blur)
        # cv.drawContours(contours_filtered_img, contours_filtered, -1, (255, 255, 255), -1)

        # blur_scharr = cv.GaussianBlur(bgr, (blur_val, blur_val), 0)
        # sharpen_scharr = cv.filter2D(blur_scharr, -1, np.array([[-1, -1, -1], [-1, sharpen_val, -1], [-1, -1, -1]]))
        # scharr_edges = scharr_edge_detection(sharpen_scharr)

        # segmented_image = get_kmeans(blur_scharr, 7)
        # cv.imshow("blur", blur)
        # cv.imshow("canny", canny_dilate)
        # cv.imshow("canny_drawn", canny_drawn)
        # cv.imshow("contours filtered", contours_filtered_img)
        # cv.imshow("canny sharpen inverted", canny_sharpen_inverted)
        # cv.imshow("sharpen", sharpen)
        # cv.imshow("sharpen_inverted", sharpen_inverted)
        # cv.imshow("scharr", scharr_edges)
        # cv.imshow("frame", bgr)
        # cv.imshow("segmented image", segmented_image)

        key = cv.waitKey(20)
        if key == ord('c'):
            cv.imwrite('{}_{}.{}'.format(base_path, str(frame_num).zfill(digit), 'png'), canny)
            print("Screenshot saved!")
            frame_num += 1
        if key == 27:  # exit on ESC
            cam.release()
            cv.destroyAllWindows()
            break


canny_sliders()


