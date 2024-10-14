import copy
import math

import cv2 as cv
import numpy as np
from collections import namedtuple

Ellipse = namedtuple("ELLIPSE", ["x", "y", "a", "b", "angle"])


def ellipse2circle(Ellipse):
    angle = (Ellipse.angle) * math.pi / 180
    x = Ellipse.x
    y = Ellipse.y
    a = Ellipse.a
    b = Ellipse.b

    # build transformation matrix http://math.stackexchange.com/questions/619037/circle-affine-transformation
    r1 = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    r2 = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

    t1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
    t2 = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])

    d = np.array([[1, 0, 0], [0, a / b, 0], [0, 0, 1]])

    m = t2.dot(r2.dot(d.dot(r1.dot(t1))))

    return m

def get_bright_fields(frame):
    pass

def find_dartboard_outer_ellipse(frame):

    # Apply a little more contrast
    # alpha = 1  # contrast
    # beta = -60  # brightness
    # frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # Blur the image
    blur = cv.blur(frame, (5, 5))

    # img_saved = copy.deepcopy(frame)
    # cv.imshow("find ellipse here", img_saved)

    # Create a mask that finds colors in the blurred image
    lower = np.array([0, 97, 76])
    upper = np.array([179, 255, 255])
    # Create HSV Image and threshold into a range.
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    mask_color = cv.inRange(hsv, lower, upper)

    # Find contours and apply them for a complete outer ring contour
    contours, hierarchy = cv.findContours(mask_color, 1, 2)
    # contours_applied = cv.drawContours(mask_color, contours, -1, (255, 255, 255), 13)
    contours_applied = cv.drawContours(mask_color, contours, -1, (255, 255, 255), 2)


    # Floodfill from point 0, 0 with white
    # h, w = contours_applied.shape[:2]
    # floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
    # cv.floodFill(contours_applied, floodfill_mask, (0, 0), 255)

    # Find contours in the image again to find the ellipse
    contours, hierarchy = cv.findContours(contours_applied, 1, 2)

    # Define a minimum and maximum contour area
    min_thresh_e = 50000
    max_thresh_e = 250000

    # contourArea threshold important -> make accessible
    ellipse_contours = []
    for cnt in contours:
        try:
            if min_thresh_e < cv.contourArea(cnt) < max_thresh_e:

                ellipse_contours.append(cnt)

        except Exception as e:
            print("Something went wrong while finding ellipse: " + str(e))

    if not ellipse_contours:
        print("Ellipse contour not found!")
        return None, None
    biggest_contour = max(ellipse_contours, key=cv.contourArea)
    ellipse = cv.fitEllipse(biggest_contour)
    # cv.ellipse(frame, ellipse, (0, 255, 0), 2)

    x, y = ellipse[0]
    y += 5  # move ellipse a bit down
    a, b = ellipse[1]
    angle = ellipse[2]

    a = a / 2
    b = b / 2

    return Ellipse(x, y, a, b, angle), biggest_contour


def get_canny(frame, canny_lower, canny_upper):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(blur, canny_lower, canny_upper)
    return canny


def nothing(x):
    pass


def find_lines(canny_img, threshold, min_line_length, max_line_gap):
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # canny_thicker = cv.dilate(canny_img, kernel)

    # Apply HoughLinesP method to directly obtain line end points
    # lines_list = []
    lines = cv.HoughLinesP(
        canny_img,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=threshold,  # Min number of votes for valid line
        minLineLength=min_line_length,  # Min allowed length of line
        maxLineGap=max_line_gap  # Max allowed gap between line for joining them
    )

    return lines


def get_transformation_points(preview_name, cam_id):
    cam = cv.VideoCapture(cam_id)
    # cv.namedWindow(preview_name)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    cv.namedWindow('canny')
    cv.createTrackbar('canny lower', 'canny', 0, 255, nothing)
    cv.createTrackbar('canny upper', 'canny', 0, 255, nothing)
    cv.setTrackbarPos('canny lower', 'canny', 254)
    cv.setTrackbarPos('canny upper', 'canny', 255)

    cv.namedWindow('lines')
    cv.createTrackbar('threshold', 'lines', 0, 300, nothing)
    cv.createTrackbar('minLineLength', 'lines', 0, 800, nothing)
    cv.createTrackbar('maxLineGap', 'lines', 0, 200, nothing)
    cv.setTrackbarPos('threshold', 'lines', 90)
    cv.setTrackbarPos('minLineLength', 'lines', 170)
    cv.setTrackbarPos('maxLineGap', 'lines', 50)

    while rval:
        rval, frame = cam.read()

        # find enclosing ellipse
        Ellipse, contour = find_dartboard_outer_ellipse(frame)

        if Ellipse is not None:
            ellipse_mask = cv.ellipse(np.zeros(frame.shape[:2], dtype="uint8"), (int(Ellipse.x), int(Ellipse.y)), (int(Ellipse.a), int(Ellipse.b)), int(Ellipse.angle), 0.0, 360.0, color=(255, 255, 255), thickness=-1)
            # cv.imshow("ellipse_mask", ellipse_mask)
            ellipse_mask_applied = cv.bitwise_and(frame, frame, mask=ellipse_mask)
            # cv.imshow("ellipse_mask_applied", ellipse_mask_applied)
            # warp_perspective(Ellipse, contour, frame)



            canny_lower = cv.getTrackbarPos('canny lower', 'canny')
            canny_upper = cv.getTrackbarPos('canny upper', 'canny')
            canny_img = get_canny(ellipse_mask_applied, canny_lower, canny_upper)
            # canny_img = get_canny(ellipse_mask_applied, 50, 120)
            cv.imshow('canny', canny_img)

            threshold = cv.getTrackbarPos('threshold', 'lines')
            min_line_length = cv.getTrackbarPos('minLineLength', 'lines')
            max_line_gap = cv.getTrackbarPos('maxLineGap', 'lines')
            lines = find_lines(canny_img, threshold, min_line_length, max_line_gap)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv.line(ellipse_mask_applied, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.imshow('lines', ellipse_mask_applied)

        # cv.imshow(preview_name, frame)
        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            cam.release()
            cv.destroyAllWindows()
            break


def warp_perspective(Ellipse, contour, frame):
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(frame, [box], 0, (0, 0, 255), 2)
    pt_C = (box[0][0], box[0][1])
    pt_D = (box[1][0], box[1][1])
    pt_A = (box[2][0], box[2][1])
    pt_B = (box[3][0], box[3][1])
    # Here, I have used L2 norm. You can use L1 also.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    # output_pts = np.float32([[0, 0],
    #                          [0, maxHeight - 1],
    #                          [maxWidth - 1, maxHeight - 1],
    #                          [maxWidth - 1, 0]])
    output_pts = np.float32([[0, 0],
                             [0, 600 - 1],
                             [600 - 1, 600 - 1],
                             [600 - 1, 0]])
    # transformation_matrix = cv.getPerspectiveTransform(input_pts, output_pts)
    transformation_matrix = ellipse2circle(Ellipse)
    # out = cv.warpPerspective(frame, transformation_matrix, (maxWidth, maxHeight), flags=cv.INTER_LINEAR)
    out = cv.warpPerspective(frame, transformation_matrix, (600, 600), flags=cv.INTER_LINEAR)
    cv.imshow("Transformed Image", out)


# get_transformation_points("Ellipse", 0)