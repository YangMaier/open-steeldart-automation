import os
import pathlib

import cv2 as cv
import numpy as np

from operations_on.contours import get_center_of_contour_normalized
import personal_settings
from fiddling.misc import extracted_contours


def get_center_of_contour(contour):
    m = cv.moments(contour)
    try:
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
    except ZeroDivisionError as zde:
        cx = contour[0][0][0]
        cy = contour[0][0][1]
    return cx, cy


def filter_center_inside_contour(contour, point, filter_its_center):
    if not filter_its_center:
        return True
    else:
        dist = cv.pointPolygonTest(contour, point, measureDist=False)
        return dist >= 0  # contour center is inside the contour


def filter_min_size_contour(contour, cnt_min_area):
    return cv.contourArea(contour) > cnt_min_area


def filter_max_size_contour(contour, cnt_max_area):
    return cv.contourArea(contour) < cnt_max_area


def filter_contour_solidity(contour, solidity_threshold=70):
    # Solidity is the ratio of contour area to its convex hull area
    solidity_threshold = solidity_threshold * 0.01
    area = cv.contourArea(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    if hull_area == 0:
        solidity = 0
    else:
        solidity = float(area) / hull_area
    return solidity > solidity_threshold


def filter_contour_extend(contour, extend_threshold):
    # Extend is the ratio of contour area to its bounding rectangle area
    extend_threshold = extend_threshold * 0.01
    area = cv.contourArea(contour)
    x, y, w, h = cv.boundingRect(contour)
    rect_area = w * h
    if rect_area == 0:
        extend = 0
    else:
        extend = float(area) / rect_area
    return extend > extend_threshold

def matches_double_field_contour(contour, match_threshold=100):
    match_threshold = match_threshold * 0.01
    ret = cv.matchShapes(contour, extracted_contours.double_field_contour, 2, 0.0)
    print(round(ret, 4))

    return ret < match_threshold, round(ret, 4)


def is_double_field(contour, cnt_min_area, cnt_max_area, cnt_min_solidity, cnt_matches_double_field, cnt_center_inside_contour):
    center = get_center_of_contour(contour)
    c1 = filter_min_size_contour(contour, cnt_min_area)
    c2 = filter_max_size_contour(contour, cnt_max_area)
    c3 = filter_contour_solidity(contour, cnt_min_solidity)
    c4, _ = matches_double_field_contour(contour, cnt_matches_double_field)
    c5 = filter_center_inside_contour(contour, center, cnt_center_inside_contour)

    # print(c1, c2, c3, c4, c5)

    return c1 and c2 and c3 and c4 and c5


def is_elongated(cnt, length_ratio):
    rect = cv.minAreaRect(cnt)
    # rect[0] is center x,y
    # rect[1] is width and height
    if min(rect[1]) <= 0:
        return False  # Avoid division by zero. Is not a double field anyway with height or width 0
    if (max(rect[1]) / min(rect[1])) > (length_ratio / 100):
        return True
    else:
        return False


def get_longer_rect(cnt):
    (x, y), (w, h), angle = cv.minAreaRect(cnt)
    length_ratio = 1.2
    if w != 0 and h != 0:
        if w > h:
            w *= length_ratio
        if h > w:
            h *= length_ratio
    box = cv.boxPoints(((x, y), (w, h), angle))
    box = np.int0(box)
    return box


def get_sift_img(thresh_img):
    sift = cv.SIFT.create()
    kp = sift.detect(thresh_img, None)
    empty_img = np.zeros(thresh_img.shape[:2], dtype="uint8")
    img = cv.drawKeypoints(empty_img, kp, empty_img)
    return img


def cam_preview_parameters(preview_name, cam_id):
    """
        Displays a preview of the specified webcam.

        Args:
            preview_name (str): The name of the window to display the preview in.
            cam_id (int): The ID of the webcam to preview. Defaults to 0.

        Returns:
            None

        Description:
            This function opens a window and displays a preview of the specified webcam.
            The `cam_id` parameter specifies the ID of the webcam to preview. If `cam_id` is not specified,
            the function defaults to previewing the first webcam. The function will continue to display the preview
            window until the user closes it.

        Example:
            cam_preview(1)
        """
    cam = cv.VideoCapture(cam_id)
    # cam.set(cv.CAP_PROP_FRAME_WIDTH, personal_settings.live_img_width)
    # cam.set(cv.CAP_PROP_FRAME_HEIGHT, personal_settings.live_img_height)
    # cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    # cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    def nothing(x):
        pass

    # Create a window
    cv.namedWindow('image')
    cv.namedWindow('contour filtering')

    # create trackbars for color change
    cv.createTrackbar('saturation offset', 'image', 0, 150, nothing)
    cv.createTrackbar('HMin Part 1', 'image', 0, 179, nothing)  # Hue is from 0-179 for Opencv
    cv.createTrackbar('HMin Part 2', 'image', 0, 179, nothing)  # Hue is from 0-179 for Opencv
    cv.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv.createTrackbar('HMax Part 1', 'image', 0, 179, nothing)
    cv.createTrackbar('HMax Part 2', 'image', 0, 179, nothing)
    cv.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv.createTrackbar('VMax', 'image', 0, 255, nothing)

    # create trackbars for shape filtering
    cv.createTrackbar('cnt min area', 'contour filtering', 0, 1000, nothing)
    cv.createTrackbar('cnt max area', 'contour filtering', 0, 10000, nothing)
    # cv.createTrackbar('cnt min solidity', 'contour filtering', 0, 100, nothing)
    cv.createTrackbar('cnt min extent', 'contour filtering', 0, 100, nothing)
    # cv.createTrackbar('cnt matches double field', 'contour filtering', 0, 10000, nothing)
    cv.createTrackbar('cnt length ratio', 'contour filtering', 0, 800, nothing)
    cv.createTrackbar('cnt center inside contour', 'contour filtering', 0, 1, nothing)

    # Set default value for MAX HSV trackbars.
    cv.setTrackbarPos('saturation offset', 'image', 0)
    cv.setTrackbarPos('SMin', 'image', 120)
    cv.setTrackbarPos('VMin', 'image', 86)
    cv.setTrackbarPos('HMax Part 1', 'image', 179)
    cv.setTrackbarPos('HMax Part 2', 'image', 179)
    cv.setTrackbarPos('SMax', 'image', 255)
    cv.setTrackbarPos('VMax', 'image', 255)

    # set default values for shape filter trackbars
    cv.setTrackbarPos('cnt min area', 'contour filtering', 170)
    cv.setTrackbarPos('cnt max area', 'contour filtering', 1200)
    # cv.setTrackbarPos('cnt min solidity', 'contour filtering', 70)
    cv.setTrackbarPos('cnt min extent', 'contour filtering', 17)
    # cv.setTrackbarPos('cnt matches double field', 'contour filtering', 120)
    cv.setTrackbarPos('cnt length ratio', 'contour filtering', 300)
    cv.setTrackbarPos('cnt center inside contour', 'contour filtering', 0)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while rval:
        # frame = cv.bitwise_not(frame)  # invert color
        # alpha = 1  # contrast
        # beta = -60  # brightness
        # frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
        blur = cv.blur(frame, (5, 5))
        cv.imshow("Frame", blur)

        ycrcb_img = cv.cvtColor(blur, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(ycrcb_img)
        y_equalized = cv.equalizeHist(y)
        ycrcb = cv.merge((y_equalized, cr, cb))
        equalized_image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

        # Create HSV Image and threshold into a range.
        hsv = cv.cvtColor(equalized_image, cv.COLOR_BGR2HSV)

        # get current positions of all trackbars
        saturation_offset = cv.getTrackbarPos('saturation offset', 'image')
        hMin1 = cv.getTrackbarPos('HMin Part 1', 'image')
        hMin2 = cv.getTrackbarPos('HMin Part 2', 'image')
        sMin = cv.getTrackbarPos('SMin', 'image')
        vMin = cv.getTrackbarPos('VMin', 'image')

        hMax1 = cv.getTrackbarPos('HMax Part 1', 'image')
        hMax2 = cv.getTrackbarPos('HMax Part 2', 'image')
        sMax = cv.getTrackbarPos('SMax', 'image')
        vMax = cv.getTrackbarPos('VMax', 'image')

        (h, s, v) = cv.split(hsv)
        s = s + saturation_offset
        s = np.clip(s, 0, 255)
        hsv = cv.merge([h, s, v])

        blur = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        # Set minimum and max HSV values to display
        lower1 = np.array([hMin1, sMin, vMin])
        upper1 = np.array([hMax1, sMax, vMax])

        lower2 = np.array([hMin2, sMin, vMin])
        upper2 = np.array([hMax2, sMax, vMax])

        mask1 = cv.inRange(hsv, lower1, upper1)
        mask2 = cv.inRange(hsv, lower2, upper2)

        mask = cv.bitwise_or(mask1, mask2)

        masked_frame = cv.bitwise_and(frame, frame, mask=mask)

        cv.imshow("Masked Frame", masked_frame)

        # gray = cv.cvtColor(blur, cv.COLOR_BGR2)
        # blur = cv.GaussianBlur(gray, (3, 3), 0)
        cv.imshow("Gray and Blur", blur)

        # kernel = np.ones((3, 3), np.uint8)
        # opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
        cv.imshow('image', mask)

        contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        cnt_min_area = cv.getTrackbarPos('cnt min area', 'contour filtering')
        cnt_max_area = cv.getTrackbarPos('cnt max area', 'contour filtering')
        # cnt_min_solidity = cv.getTrackbarPos('cnt min solidity', 'contour filtering')
        cnt_min_extent = cv.getTrackbarPos('cnt min extent', 'contour filtering')
        # cnt_matches_double_field = cv.getTrackbarPos('cnt matches double field', 'contour filtering')
        length_ratio = cv.getTrackbarPos('cnt length ratio', 'contour filtering')
        cnt_center_inside_contour = cv.getTrackbarPos('cnt center inside contour', 'contour filtering')

        double_contours = []
        for i, cnt in enumerate(contours):
            center = get_center_of_contour(cnt)
            c1 = filter_min_size_contour(cnt, cnt_min_area)
            c2 = filter_max_size_contour(cnt, cnt_max_area)
            c3 = filter_contour_extend(cnt, cnt_min_extent)
            c4 = is_elongated(cnt, length_ratio)
            # c4, match = matches_double_field_contour(cnt, cnt_matches_double_field)
            c5 = filter_center_inside_contour(cnt, center, cnt_center_inside_contour)
            # empty_img = np.zeros(frame.shape[:2], dtype="uint8")

            # window_title = str(i) + " " + str(c1) + " " + str(c2) + " " + str(c3) + " " + str(match) + " " + str(c5)
            # cv.drawContours(empty_img, [cnt], -1, (255, 255, 255), thickness=cv.FILLED)
            # cv.imshow(window_title, empty_img)
            if c1 and c2 and c3 and c4 and c5:
                double_contours.append(cnt)
                double_contours.append(get_longer_rect(cnt))

        # cv.waitKey(0)
        # empty_img = np.zeros(frame.shape[:2], dtype="uint8")
        # double_contours.append(extracted_contours.double_field_contour)
        # find_ellipse_here = cv.drawContours(empty_img, double_contours, -1, (255, 255, 255), thickness=cv.FILLED)



        # cv.imshow('contour filtering', find_ellipse_here)

        # import high_contrast_image
        # high_contrast = high_contrast_image.high_contrast_image(blur)
        # cv.imshow("High Contrast", high_contrast)

        # brightness_mask = mask.brightness_mask(frame, 225)
        # masked_image = cv.bitwise_and(blur, blur, mask=brightness_mask)
        # cv.imshow("Mask Brightness", brightness_mask)

        # # filter out stuff that is too dark
        # threshold = 100
        # assignvalue = 255  # Value to assign the pixel if the threshold is met
        # _, threshold_black_img = cv.threshold(blur, threshold, assignvalue, cv.THRESH_BINARY)
        # cv.imshow("Threshold Dark", threshold_black_img)
        #
        # threshold_black_img = cv.bitwise_not(threshold_black_img)  # invert color
        #
        # # filter out stuff that is too bright
        # threshold = 200
        # assignvalue = 255  # Value to assign the pixel if the threshold is met
        # _, threshold_white_img = cv.threshold(threshold_black_img, threshold, assignvalue, cv.THRESH_BINARY)
        # cv.imshow("Threshold Bright", threshold_white_img)

        # threshold_white_img = cv.bitwise_not(frame)  # invert color

        # apply automatic Canny edge detection using the computed median
        # median_with_fixed_offset = np.median(empty_img) + 40
        # sigma = 0.33
        # lower_t = int(max(0, (1.0 - sigma) * median_with_fixed_offset))
        # upper_t = int(min(255, (1.0 + sigma) * median_with_fixed_offset))
        # canny = cv.Canny(empty_img, lower_t, upper_t)
        # cv.imshow("Auto Canny", canny)

        frame_alpha = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
        masked_alpha = cv.bitwise_and(frame_alpha, frame_alpha, mask=mask)

        pixel_count = frame.shape[0] * frame.shape[1]
        key = cv.waitKey(20)
        if key == ord('s'):
            # save all contours separately
            folder = pathlib.Path().joinpath("../../src/media/contours")
            os.makedirs(folder, exist_ok=True)
            for cnt in contours:
                contour_area = cv.contourArea(cnt)
                if contour_area > 500:
                    test_img = np.zeros(frame.shape[:2], dtype="uint8")
                    cv.drawContours(test_img, [cnt], -1, (255, 255, 255), thickness=cv.FILLED)
                    x, y, w, h = cv.boundingRect(cnt)  # xy is top left corner
                    contour_area_normalized = round(contour_area / pixel_count, 5)
                    cp_normalized = get_center_of_contour_normalized(cnt, personal_settings.live_img_width, personal_settings.live_img_height)
                    filename = f"contour_{cp_normalized.x}_{cp_normalized.y}_{contour_area_normalized}.png"
                    filename_img = f"contour_{cp_normalized.x}_{cp_normalized.y}_{contour_area_normalized}_img.png"
                    contour_file_path = folder.joinpath(filename)
                    contour_img_file_path = folder.joinpath(filename_img)
                    img_crop = test_img[
                            max(y-1, 0):min(y + h + 1, personal_settings.live_img_height),
                            max(x-1, 0):min(x + w + 1, personal_settings.live_img_width)
                        ]
                    try:
                        cv.imwrite(str(contour_file_path), img_crop)
                        cv.imwrite(str(contour_img_file_path), test_img)
                    except Exception as e:
                        print(f"Could not write img with filename: {(contour_file_path)}, witdh: {w} and height: {h}")
                        print(f"Reason: {e}")

        if key == 27:  # exit on ESC
            break

        rval, frame = cam.read()

    cam.release()
    cv.destroyAllWindows()

cam_preview_parameters('Preview', 2)