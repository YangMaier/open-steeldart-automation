import copy

import cv2 as cv
import numpy as np

import src.data_structures.board_segment_old.ellipse


def find_ellipse_once(frame):
    # Apply a little more contrast
    alpha = 1  # contrast
    beta = -60  # brightness
    frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # Blur the image
    blur = cv.blur(frame, (3, 3))

    # Create a mask that finds colors in the blurred image
    lower = np.array([0, 137, 76])
    upper = np.array([179, 255, 255])
    # Create HSV Image and threshold into a range.
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    mask_color = cv.inRange(hsv, lower, upper)

    # Find contours and apply them for a complete outer ring contour
    contours, hierarchy = cv.findContours(mask_color, 1, 2)
    contours_applied = cv.drawContours(mask_color, contours, -1, (255, 255, 255), 5)

    # Floodfill from point 0, 0 with white
    h, w = contours_applied.shape[:2]
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(contours_applied, floodfill_mask, (0, 0), 255)

    # Find contours in the image again to find the ellipse
    contours, hierarchy = cv.findContours(contours_applied, 1, 2)

    # Define a minimum and maximum contour area
    min_thresh_e = 50000
    max_thresh_e = 250000

    # contourArea threshold important -> make accessible
    for cnt in contours:
        try:
            if min_thresh_e < cv.contourArea(cnt) < max_thresh_e:
                ellipse = cv.fitEllipse(cnt)
                cv.ellipse(frame, ellipse, (0, 255, 0), 2)

                x, y = ellipse[0]
                a, b = ellipse[1]
                angle = ellipse[2]

                a = a / 2
                b = b / 2

                return src.data_structures.ellipse.Ellipse(x, y, a, b, angle)

        except Exception as e:
            print("Something went wrong while finding ellipse: " + str(e))


def find_ellipse_custom():
    cam = cv.VideoCapture(0)

    if not cam.isOpened():  # try to get the first frame
        print("Cam " + str(0) + " could not be opened.")

    rval, frame = cam.read()
    while rval:

        rval, frame = cam.read()
        key = cv.waitKey(20)
        # color invert test
        # frame = cv.bitwise_not(frame)

        alpha = 1  # contrast
        beta = -60  # brightness
        frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)

        blur = cv.blur(frame, (3, 3))

        # Set minimum and max HSV values to display
        lower = np.array([0, 137, 76])
        upper = np.array([179, 255, 255])
        # Create HSV Image and threshold into a range.
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        mask_color = cv.inRange(hsv, lower, upper)

        # mask_color = cv.bitwise_not(mask_color)

        saved_variant = copy.deepcopy(mask_color)

        contours, hierarchy = cv.findContours(mask_color, 1, 2)

        # contours_applied = cv.drawContours(mask_color, contours, -1, (0, 0, 0), 5)
        contours_applied = cv.drawContours(mask_color, contours, -1, (255, 255, 255), 5)

        # Get mask for floodfill
        h, w = contours_applied.shape[:2]
        floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)



        # Floodfill from point (0, 0)
        cv.floodFill(contours_applied, floodfill_mask, (0, 0), 255)

        contours, hierarchy = cv.findContours(contours_applied, 1, 2)

        # min_thresh_e = 200000/4
        # max_thresh_e = 1000000/4
        min_thresh_e = 50000
        max_thresh_e = 250000

        # contourArea threshold important -> make accessible
        for cnt in contours:
            try:  # threshold critical, change on demand?
                if min_thresh_e < cv.contourArea(cnt) < max_thresh_e:
                    ellipse = cv.fitEllipse(cnt)
                    cv.ellipse(frame, ellipse, (0, 255, 0), 2)

                    x, y = ellipse[0]
                    a, b = ellipse[1]
                    angle = ellipse[2]

                    center_ellipse = (x, y)

                    a = a / 2
                    b = b / 2

                    cv.ellipse(frame, (int(x), int(y)), (int(a), int(b)), int(angle), 0.0, 360.0, (255, 0, 0))
                    cv.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

                    # return a, b, x, y, angle, center_ellipse
            # corrupted file
            except Exception as e:
                print("Error in find_ellipse_custom: " + str(e))

        cv.imshow("Frame", frame)
        cv.imshow("Mask", saved_variant)
        cv.imshow("Masked", mask_color)

        if key == 27:  # exit on ESC
            cv.destroyAllWindows()
            cam.release()
            return 0
        if key == 32:  # next params on SPACE
            cv.destroyAllWindows()
            break


def draw_ellipse(_image, ellipse):
    x = ellipse[0]
    y = ellipse[1]
    a = ellipse[2]
    b = ellipse[3]
    angle = ellipse[4]
    cv.ellipse(_image, (int(x), int(y)), (int(a), int(b)), int(angle), 0.0, 360.0, (255, 0, 0))


def find_ellipses_ximgproc(masked_image, _processed_image, s_t, r_t, c_t, ellipse_was_found_one_time):
    # apply automatic Canny edge detection using the computed median
    # median_with_fixed_offset = np.median(masked_image) + 40
    # sigma = 0.33
    # lower_t = int(max(0, (1.0 - sigma) * median_with_fixed_offset))
    # upper_t = int(min(255, (1.0 + sigma) * median_with_fixed_offset))
    # canny = cv.Canny(masked_image, lower_t, upper_t)
    #
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # canny_thicker = cv.dilate(canny, kernel)

    # find_ellipse_custom(canny_thicker, frame)
    find_ellipses_ximgproc(masked_image, _processed_image, s_t, r_t, c_t)

    window_name = "s: " + str(s_t) + " r: " + str(r_t) + " c: " + str(c_t)
    cv.namedWindow(window_name)

    cv.imshow(window_name, _processed_image)
    cv.imshow("Canny", masked_image)

    ellipses = cv.ximgproc.findEllipses(
        masked_image,
        scoreThreshold=s_t,
        reliabilityThreshold=r_t,
        centerDistanceThreshold=c_t
    )
    if ellipses is not None:
        for ell in ellipses:
            draw_ellipse(_processed_image, ell[0])
        return True
    else:
        return False


def find_stuff():
    cam = cv.VideoCapture(0)

    if not cam.isOpened():  # try to get the first frame
        print("Cam " + str(0) + " could not be opened.")

    else:
        s_t = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        r_t = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]  # 0.1 was useless all the time
        c_t = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        for s, r, c in ((s, r, c) for s in s_t for r in r_t for c in c_t):  # all combinations of sT, rT and cT

            rval, frame = cam.read()
            ellipse_was_found_one_time = False

            while rval:
                rval, frame = cam.read()
                key = cv.waitKey(20)
                alpha = 1  # contrast
                beta = -60  # brightness
                frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)

                blur = cv.blur(frame, (3, 3))

                # Set minimum and max HSV values to display
                lower = np.array([0, 137, 76])
                upper = np.array([179, 255, 255])
                # Create HSV Image and threshold into a range.
                hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
                mask_color = cv.inRange(hsv, lower, upper)

                contours, hierarchy = cv.findContours(mask_color, 1, 2)

                contours_applied = cv.drawContours(mask_color, contours, -1, (255, 255, 255), 5)

                # Get mask for floodfill
                h, w = contours_applied.shape[:2]
                floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)

                # Floodfill from point (0, 0)
                cv.floodFill(contours_applied, floodfill_mask, (0, 0), 255)

                contours_applied = cv.bitwise_not(contours_applied)  # invert color

                window_name = "s: " + str(s) + " r: " + str(r) + " c: " + str(c)
                cv.namedWindow(window_name)

                cv.imshow("Floodfilled Image", contours_applied)

                ellipses = cv.ximgproc.findEllipses(
                    contours_applied,
                    scoreThreshold=s,
                    reliabilityThreshold=r,
                    centerDistanceThreshold=c
                )
                if ellipses is not None:
                    for ell in ellipses:
                        draw_ellipse(frame, ell[0])

                cv.imshow(window_name, frame)

                # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # blur = cv.GaussianBlur(masked_red_and_green, (3, 3), 0)

                if key == 27:  # exit on ESC
                    cv.destroyAllWindows()
                    cam.release()
                    return 0
                if key == 32:  # next params on SPACE
                    cv.destroyAllWindows()
                    break