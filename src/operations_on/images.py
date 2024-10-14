import logging
import time
from typing import List

import cv2 as cv
import numpy as np

from skimage.metrics import structural_similarity

from data_structures.coordinates_extended import RadialSectionRingIntersections, RadialSectionRingIntersectionsEllipse, \
    CoordinateAngleDistance
from data_structures.ellipse import Ellipse
from data_structures.coordinates import Coordinate2d
from operations_on.contours import get_contour_distance_to, get_center_of_contour
from operations_on.cads import get_adjacent_angle_cad
from operations_on.coordinates import  get_line_extension_end_point, line_line_intersection
from operations_on.coordinates_and_angles import get_cads
from preset_circled_board_coordinates2 import __DartboardDefinition


def draw_contours(frame, contours, color, thickness=1):
    cv.drawContours(frame, contours, -1, color, thickness)

    return contours


def draw_circle(frame, point: Coordinate2d, color, radius=3):
    cv.circle(frame, (point.x, point.y), radius, color, -1)


def draw_ellipse(frame, ellipse: Ellipse, color=(255, 255, 0), thickness=1):
    cv.ellipse(frame, (ellipse.x, ellipse.y), (ellipse.a, ellipse.b), ellipse.angle, 0, 360, color=color, thickness=thickness)


def get_board_points(dart_board):
    """ Extracts the corner points of the dart board

     Extracts all number segment corner points from the associated number segments from mid to outer points clockwise

    Args:
        dart_board:

    Returns:

    """
    board_points = []
    letter_sequence_clockwise = [1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20]
    for number in letter_sequence_clockwise:
        number_segment = [ns for ns in dart_board.associated_number_segments if ns.associated_number == number][0]
        board_points.append(number_segment.inner_segment.score_border.c_1.as_np_arr())
        board_points.append(number_segment.inner_segment.score_border.c_2.as_np_arr())
        board_points.append(number_segment.outer_segment.score_border.c_1.as_np_arr())
        board_points.append(number_segment.outer_segment.score_border.c_2.as_np_arr())
        board_points.append(number_segment.double_segment.score_border.c_2.as_np_arr())
    src_pts = np.asarray(board_points)
    return src_pts


def get_board_points_clockwise(radial_section_ring_intersections: List[RadialSectionRingIntersections]):
    letter_sequence_clockwise = [1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20]
    board_coordinates = []
    for number in letter_sequence_clockwise:
        letter_corners_radial_section = [corner for corner in radial_section_ring_intersections if corner.number == number][0]
        letter_corners = letter_corners_radial_section.get_by_distance()
        board_coordinates.append(letter_corners)


def get_homography(src_pts: np.ndarray, reference_points: np.ndarray):
    homography_matrix, status = cv.findHomography(src_pts, reference_points)
    return homography_matrix


def transform_img(img_bgr, src_pts: np.ndarray, reference_points: np.ndarray, new_dims):
    """ Rotates and warps the image to fit a picture of a dart board that matches the reference_points as corners
    Can be used to warp and rotate the image to a view that shows the dartboard as viewed from the front
    If new_dims is given, warps the image to the new dimensions instead

    Args:
        img_bgr:
        src_pts:
        reference_points:
        new_dims:

    Returns: bgr img with new_dims as dimensions

    """
    H = get_homography(src_pts, reference_points)

    img_warped = cv.warpPerspective(img_bgr, H, new_dims)

    return img_warped


def get_img_diff_skikit_similarity(base_frame, frame):
    grey_base = cv.cvtColor(base_frame, cv.COLOR_BGR2GRAY)
    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(grey_base, grey_frame, full=True)
    diff = (diff * 255).astype("uint8")
    diff = cv.bitwise_not(diff)
    return diff


# https://sokacoding.medium.com/simple-motion-detection-with-python-and-opencv-for-beginners-cdd4579b2319
def motion_is_detected_mean(frame, last_mean):
    """
    Looks like this is not working as intended in the current form
    Args:
        frame:
        last_mean:

    Returns:

    """
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mean = np.mean(grey)
    result = round(np.abs(mean - last_mean), 4)
    print(result)
    if result > 0.4:
        return True, mean
    else:
        return False, mean


def motion_is_detected_slow(last_frame, frame):
    img_diff = cv.absdiff(last_frame, frame)
    grey = cv.cvtColor(img_diff, cv.COLOR_BGR2GRAY)
    thresh_diff = cv.threshold(grey, 8, 255, cv.THRESH_BINARY)[1]
    contours, _ = cv.findContours(thresh_diff, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if img_diff.shape[0] == 1080 or img_diff.shape[0] == 1920:
        contour_area_filter = 800
    elif img_diff.shape[0] == 640 or img_diff.shape[0] == 480:
        contour_area_filter = 200
    else:
        raise NotImplementedError
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > contour_area_filter]
    if not contours:
        return False
    else:
        return True


def get_masked_diff(base_frame, frame):
    img_diff = cv.absdiff(base_frame, frame)

    grey = cv.cvtColor(img_diff, cv.COLOR_BGR2GRAY)

    thresh_diff = cv.threshold(grey, 8, 255, cv.THRESH_BINARY)[1]

    contours, _ = cv.findContours(thresh_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if img_diff.shape[0] == 1080 or img_diff.shape[0] == 1920:
        contour_area_filter = 40
    elif img_diff.shape[0] == 640 or img_diff.shape[0] == 480:
        contour_area_filter = 10
    else:
        raise NotImplementedError

    # filter contours by size and by distance to the largest contour (the dart scheme/contour we want to track)
    contour_sizes = [(cv.contourArea(cnt), cnt) for cnt in contours]
    contour_sizes.sort(key=lambda x: x[0], reverse=True)
    dart_contour = contour_sizes[0][1]
    other_contours = [cnt_size[1] for cnt_size in contour_sizes[1:]]
    other_contours_filtered = [cnt for cnt in other_contours if cv.contourArea(cnt) > contour_area_filter]
    other_contours_distances = [
        (
            get_contour_distance_to(dart_contour, get_center_of_contour(cnt)),
            cnt
        )
        for cnt in other_contours_filtered
    ]
    other_contours_filtered = [dist_cnt[1] for dist_cnt in other_contours_distances if dist_cnt[0] > -50]
    other_contours_filtered.append(dart_contour)

    img_mask = np.zeros(grey.shape[:2], dtype="uint8")
    cv.drawContours(img_mask, other_contours_filtered, -1, 255, -1)
    kernel_erode = np.ones((3, 3), np.uint8)
    mask_eroded = cv.erode(img_mask, kernel_erode, iterations=3)
    masked_frame = cv.bitwise_and(frame, frame, mask=mask_eroded)

    return masked_frame


def skikit_diff_dart_approx(base_frame, frame):

    img_diff = get_img_diff_skikit_similarity(base_frame, frame)

    diff_thresh_t = cv.threshold(img_diff, 25, 0, cv.THRESH_TOZERO)[1]
    contours, _ = cv.findContours(diff_thresh_t, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours_filtered = [cnt for cnt in contours if cv.contourArea(cnt) > 200]
    mask_custom = np.zeros(img_diff.shape, dtype='uint8')
    cv.drawContours(mask_custom, contours_filtered, -1, 255, -1)
    opening = cv.morphologyEx(mask_custom, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    dilate_mask = cv.dilate(opening, np.ones((30, 30), np.uint8), iterations=1)
    mask_applied = cv.bitwise_and(img_diff, img_diff, mask=dilate_mask)
    #
    # diff_thresh_t = cv.threshold(diff_inverted, 20, 0, cv.THRESH_TOZERO)[1]
    # kernel = np.ones((3, 3), np.uint8)
    # closing = cv.morphologyEx(diff_thresh_t, cv.MORPH_CLOSE, kernel, iterations=2)
    # #
    # # diff_thresh = cv.threshold(diff, 200, 255, cv.THRESH_BINARY)[1]
    # # closing = cv.morphologyEx(diff_inverted, cv.MORPH_CLOSE, kernel, iterations=3)
    # contours, _ = cv.findContours(diff_inverted, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours_filtered = [cnt for cnt in contours if cv.contourArea(cnt) > 400]
    # #
    # # all_contour_points = []
    # # for i in range(0, len(contours_filtered)):
    # #     for p in contours_filtered[i]:
    # #         all_contour_points.append(p[0])
    # #
    # # if not all_contour_points:
    # #     return None
    # #
    # # center_pt = np.array(all_contour_points).mean(axis=0)
    # # clock_ang_dist = sort_points_clockwise.ClockwiseAngleAndDistance(center_pt)
    # # all_contours_clockwise = sorted(all_contour_points, key=clock_ang_dist)
    # # all_contours_points_clockwise_cv = (np.array(all_contours_clockwise).reshape((-1, 1, 2)).astype(np.int32))
    # #
    # # epsilon = 0.0004 * cv.arcLength(all_contours_points_clockwise_cv, True)
    # # approx = cv.approxPolyDP(all_contours_points_clockwise_cv, epsilon, True)
    # mask_custom = np.zeros(base_frame_gray.shape, dtype='uint8')
    # cv.drawContours(mask_custom, contours_filtered, -1, 255, -1)
    #
    # mask_applied = cv.bitwise_and(diff_thresh_t, diff_thresh_t, mask=dilate_mask)

    return mask_applied


def skikit_similarity(base_frame, frame):

    # Convert images to grayscale
    base_frame_gray = cv.cvtColor(base_frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(base_frame_gray, frame_gray, full=True)
    # print("Image Similarity: {:.4f}%".format(score * 100))

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] base_frame we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    diff_thresh = cv.threshold(diff, 245, 255, cv.THRESH_BINARY)[1]
    diff_inverted = cv.bitwise_not(diff_thresh)
    contours, _ = cv.findContours(diff_inverted, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours_filtered = [cnt for cnt in contours if cv.contourArea(cnt) > 30]
    mask_custom = np.zeros(base_frame_gray.shape, dtype='uint8')
    cv.drawContours(mask_custom, contours_filtered, -1, 255, -1)

    diff_box = cv.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(base_frame.shape, dtype='uint8')
    filled_frame = frame.copy()

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 20:
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(base_frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv.drawContours(mask, [cnt], 0, (255, 255, 255), -1)
            cv.drawContours(filled_frame, [cnt], 0, (0, 255, 0), -1)

    cv.imshow('base_frame', base_frame)
    cv.imshow('frame', frame)
    cv.imshow('diff', diff)
    cv.imshow('diff_box', diff_box)
    cv.imshow('mask', mask)
    cv.imshow('filled frame', filled_frame)


def get_img_features(img_masked_diff):

    # Initialize ORB detector
    orb = cv.ORB.create(nfeatures=500)

    # Detect key points and compute descriptors
    keypoints, _ = orb.detectAndCompute(img_masked_diff, None)
    # for x in keypoints:
    #     print("({:.2f},{:.2f}) = size {:.2f} angle {:.2f}".format(
    #         x.pt[0], x.pt[1], x.size, x.angle))

    img_kp = cv.drawKeypoints(
        img_masked_diff,
        keypoints,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return img_kp


def warp_image_onto_corner_calculation(equalized_bgr, smoothed_outer_double_ring_contour, radial_sections_ring_intersections, board_center_coordinate):
    # fit the outer double corners onto an ellipse that was built from the smoothed outer double ring
    # get adjacent ellipse points for the double corners
    # the ellipse points are the basis for all future calculations
    # calculate corners between opposite ellipse double corners
    # warp the image by fitting the original corners onto the calculated corners
    equalized_bgr_ellipse = equalized_bgr.copy()
    cv.drawContours(equalized_bgr_ellipse, [smoothed_outer_double_ring_contour], -1, (255, 255, 0), 1)
    (x, y), (a, b), angle = cv.fitEllipse(smoothed_outer_double_ring_contour)
    ellipse_points = cv.ellipse2Poly(
        (int(x), int(y)),
        (int(a / 2), int(b / 2)),
        angle=int(angle),
        arcStart=0,
        arcEnd=360,
        delta=1
    )
    ellipse_cads = get_cads(np.asarray(ellipse_points), board_center_coordinate)
    inflated_radial_sections_ring_intersections = []
    for radial_section_ring_intersections in radial_sections_ring_intersections:
        radial_section_cads: List[CoordinateAngleDistance] = radial_section_ring_intersections.get_by_distance()
        double_corner: CoordinateAngleDistance = radial_section_cads[-1]
        adjacent_ellipse_cad: CoordinateAngleDistance = get_adjacent_angle_cad(double_corner, ellipse_cads)
        rsrie = RadialSectionRingIntersectionsEllipse(double_corner, adjacent_ellipse_cad, radial_section_cads)
        inflated_radial_sections_ring_intersections.append(rsrie)

    i = 0
    i_opposite = 10
    opposite_radial_section_pairs = []
    for j in range(10):
        opposite_radial_section_pairs.append(
            [
                inflated_radial_sections_ring_intersections[i + j],
                inflated_radial_sections_ring_intersections[i_opposite + j]
            ]
        )

    ellipse_center = get_center_of_contour(np.asarray(ellipse_points))
    cv.circle(equalized_bgr, (ellipse_center.x, ellipse_center.y), 4, (255, 255, 0), 1)

    corner_pairs = []
    for rs1, rs2 in opposite_radial_section_pairs:
        corner_pairs.append((rs1.ellipse_corner.coordinate, rs2.ellipse_corner.coordinate))

    corner_pair_pairs = [(corner_pairs[i], (corner_pairs[(i + 1) % len(corner_pairs)])) for i in range(len(corner_pairs))]

    line_line_intersections = []
    for cp1, cp2 in corner_pair_pairs:
        intersection = line_line_intersection(cp1[0], cp1[1], cp2[0], cp2[1])
        line_line_intersections.append(intersection)

    intersections_np = np.asarray([(lli.x, lli.y) for lli in line_line_intersections])

    mid = get_center_of_contour(intersections_np)
    cv.circle(equalized_bgr, (mid.x, mid.y), 4, (255, 255, 0), 1)

    for rs1, rs2 in opposite_radial_section_pairs:
        c1 = rs1.ellipse_corner
        c2 = rs2.ellipse_corner
        # the calculations are done with line extensions from mid of c1-c2 to c1, c2 respectively
        # the mid-coordinate is always the new bulls-eye coordinate
        calculate_undistorted_board_corners(c1.coordinate, mid, rs1)
        calculate_undistorted_board_corners(c2.coordinate, mid, rs2)
        color = np.random.randint(0, 255, 3).tolist()
        cv.circle(equalized_bgr, (c1.coordinate.x, c1.coordinate.y), 5, color, -1)
        cv.circle(equalized_bgr, (c2.coordinate.x, c2.coordinate.y), 5, color, -1)
        cv.line(equalized_bgr, (c1.coordinate.x, c1.coordinate.y), (c2.coordinate.x, c2.coordinate.y), color, 1)

    scr_pts = []
    dst_pts = []
    for ring_intersection in inflated_radial_sections_ring_intersections:
        scr_pts.extend(ring_intersection.get_by_distance_as_np())
        dst_pts.extend(ring_intersection.get_calculated_as_np())

    scr_pts = np.asarray(scr_pts)
    dst_pts = np.asarray(dst_pts)

    warped_img = transform_img(equalized_bgr_ellipse, scr_pts, dst_pts, (equalized_bgr.shape[1], equalized_bgr.shape[0]))
    for rs1, rs2 in opposite_radial_section_pairs:
        cv.line(warped_img, (rs1.ellipse_corner.coordinate.x, rs1.ellipse_corner.coordinate.y), (rs2.ellipse_corner.coordinate.x, rs2.ellipse_corner.coordinate.y), (255, 255, 0), 1)
        cv.line(equalized_bgr, (rs1.ellipse_corner.coordinate.x, rs1.ellipse_corner.coordinate.y), (rs2.ellipse_corner.coordinate.x, rs2.ellipse_corner.coordinate.y), (255, 255, 0), 1)

    return warped_img


def calculate_undistorted_board_corners(corner, mid, radial_section):
    c_bull_to_inner = get_line_extension_end_point(
        mid, corner, __DartboardDefinition.length_to_bull
    )
    c_inner_triple = get_line_extension_end_point(
        mid, corner, __DartboardDefinition.length_to_inner_triple
    )
    c_triple_outer = get_line_extension_end_point(
        mid, corner, __DartboardDefinition.length_to_outer_triple
    )
    c_double_inner = get_line_extension_end_point(
        mid, corner, __DartboardDefinition.length_to_double_inner
    )
    # c1 double outer is c1
    radial_section.coordinates_calculated = [c_bull_to_inner, c_inner_triple, c_triple_outer, c_double_inner, corner]
