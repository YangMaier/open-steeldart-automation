from copy import copy
from typing import List, Tuple

import cv2 as cv
import numpy as np
import skimage.util
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize

from calculate_score_board import correct_triple_and_double_contour_categories
from operations_on.angles import get_angle_diff, is_angle_in_range
from operations_on.angles_and_distances import get_interpolated_coordinates_with_smoothed_angles
from data_structures.board_segment import BoardSegment
from data_structures.coordinates_extended import CoordinateSegmentTypeNumber, CoordinateContour, \
    RadialSectionRingIntersections, CoordinateAngleDistance
from data_structures.hsv_mask_presets import hsv_mask_red, hsv_mask_green, HSVMask
from data_structures.score_segments import SegmentType, ScoreSegment
from image_masking import get_masked_img_by_hsv_values
from operations_on.cads import smooth_transition_point_groups, get_smoothed_bull_or_bulls_eye_contour, \
    get_cads_as_contour, get_cad_with_adjacent_angle
from operations_on.contours import filter_contour_solidity, filter_min_elongation, filter_min_size_contour, \
    filter_max_size_contour, filter_max_elongation, get_center_of_contour, \
    get_as_board_segments, is_elongated, get_as_board_segment, get_smoothed_ring_cads, \
    scale_contour, get_test_img_for_contour, get_cadcs_for_contours
from operations_on.coordinates import get_mid_coordinate, get_line_extension_end_point, get_coordinate_linspace, \
    get_distance
from operations_on.coordinates_and_angles import get_mid_cad, get_endpoint_range, get_cads, get_cad
from operations_on.images import transform_img, get_homography, warp_image_onto_corner_calculation
from operations_on.letters import read_numbers_and_add_numbers_to_double_segments
from preset_circled_board_coordinates2 import get_preset_board_segments, get_real_world_coordinates


def get_transition_contour_from_box(
        equalized_hsv_blurred,
        equalized_bgr_blurred,
        box_coordinates
    ) -> Tuple[np.ndarray, bool]:

    box_mask_img = np.zeros(equalized_hsv_blurred.shape[:2], dtype=np.uint8)
    box_cnt = np.array([(bc.x, bc.y) for bc in box_coordinates], dtype=np.int32)
    cv.drawContours(box_mask_img, [box_cnt], -1, (255, 255, 255), -1)
    mask_box_eq_hsv_blurred = cv.bitwise_and(equalized_hsv_blurred, equalized_hsv_blurred, mask=box_mask_img)

    low_angle_side_coordinates = get_coordinate_linspace(box_coordinates[0], box_coordinates[1], 100)
    high_angle_side_coordinates = get_coordinate_linspace(box_coordinates[2], box_coordinates[3],
                                                          100)

    low_angle_side_hsv_mask_values = [equalized_hsv_blurred[c.y][c.x] for c in low_angle_side_coordinates]
    high_angle_side_hsv_mask_values = [equalized_hsv_blurred[c.y][c.x] for c in high_angle_side_coordinates]
    low_angle_side_mean_hsv_mask_values = np.mean(low_angle_side_hsv_mask_values, axis=0)
    high_angle_side_mean_hsv_mask_values = np.mean(high_angle_side_hsv_mask_values, axis=0)

    low_angle_side_bgr_mask_values = [equalized_bgr_blurred[c.y][c.x] for c in low_angle_side_coordinates]
    low_angle_side_mean_bgr_mask_values = np.mean(low_angle_side_bgr_mask_values)
    high_angle_side_bgr_mask_values = [equalized_bgr_blurred[c.y][c.x] for c in high_angle_side_coordinates]
    high_angle_side_mean_bgr_mask_values = np.mean(high_angle_side_bgr_mask_values)

    h_low_angle_side = low_angle_side_mean_hsv_mask_values[0]
    s_low_angle_side = low_angle_side_mean_hsv_mask_values[1]
    v_low_angle_side = low_angle_side_mean_hsv_mask_values[2]
    h_high_angle_side = high_angle_side_mean_hsv_mask_values[0]
    s_high_angle_side = high_angle_side_mean_hsv_mask_values[1]
    v_high_angle_side = high_angle_side_mean_hsv_mask_values[2]
    h_range = 10
    s_range = 50
    v_range = 30

    h_min_low_angle = 0
    h_max_low_angle = int(min(h_low_angle_side + h_range, 179))
    s_min_low_angle = int(max(s_low_angle_side - s_range, 0))
    s_max_low_angle = int(min(s_low_angle_side + s_range, 255))
    v_min_low_angle = int(max(v_low_angle_side - v_range, 0))
    v_max_low_angle = int(min(v_low_angle_side + v_range, 255))
    h_min_2_low_angle = None
    h_max_2_low_angle = None

    h_min_high_angle = 0
    h_max_high_angle = int(min(h_high_angle_side + h_range, 179))
    s_min_high_angle = int(max(s_high_angle_side - s_range, 0))
    s_max_high_angle = int(min(s_high_angle_side + s_range, 255))
    v_min_high_angle = int(max(v_high_angle_side - v_range, 0))
    v_max_high_angle = int(min(v_high_angle_side + v_range, 255))
    h_min_2_high_angle = None
    h_max_2_high_angle = None

    # black can only be matched by v_max
    if low_angle_side_mean_bgr_mask_values < high_angle_side_mean_bgr_mask_values:
        # low angle side is black
        h_max_low_angle = 179
        s_min_low_angle = 0
        s_max_low_angle = 255
        v_min_low_angle = 0
        v_max_low_angle = int(v_max_low_angle)

        # high angle side is reddish
        h_min_2_high_angle = 150
        h_max_2_high_angle = 179

    else:
        # high angle side is black
        h_max_high_angle = 179
        s_min_high_angle = 0
        s_max_high_angle = 255
        v_min_high_angle = 0
        v_max_high_angle = int(v_max_high_angle)

        # low angle side is reddish
        h_min_2_low_angle = 150
        h_max_2_low_angle = 179

    mask_hsv_low_angle_side = HSVMask(h_min_low_angle, h_max_low_angle, s_min_low_angle, s_max_low_angle,
                                      v_min_low_angle, v_max_low_angle, h_min_2_low_angle, h_max_2_low_angle)
    mask_low_angle = get_masked_img_by_hsv_values(mask_box_eq_hsv_blurred, mask_hsv_low_angle_side)

    mask_hsv_high_angle_side = HSVMask(h_min_high_angle, h_max_high_angle, s_min_high_angle, s_max_high_angle,
                                       v_min_high_angle, v_max_high_angle, h_min_2_high_angle, h_max_2_high_angle)
    mask_high_angle = get_masked_img_by_hsv_values(mask_box_eq_hsv_blurred, mask_hsv_high_angle_side)

    mask_low_angle_inv = cv.bitwise_xor(box_mask_img, mask_low_angle)
    mask_high_angle_inv = cv.bitwise_xor(box_mask_img, mask_high_angle)
    side_mask = cv.bitwise_and(mask_low_angle_inv, mask_high_angle_inv)
    contours, _ = cv.findContours(side_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours_filtered = [cnt for cnt in contours if is_elongated(cnt, 4) and cv.contourArea(cnt) > 20]
    transition_contour_img = np.zeros_like(side_mask)
    cv.drawContours(transition_contour_img, contours_filtered, -1, (255, 255, 255), -1)

    test_img_ski = skimage.util.img_as_float(transition_contour_img)

    # skeleton_medial_axis = medial_axis(test_img_ski)
    # skeleton_medial_axis_cv = skimage.util.img_as_ubyte(skeleton_medial_axis)
    skeleton_transition_contour_img = skeletonize(test_img_ski)
    skeleton_cv = skimage.util.img_as_ubyte(skeleton_transition_contour_img)
    skeleton_contours, _ = cv.findContours(skeleton_cv, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if skeleton_contours:
        biggest_contour_index = np.argmax([cnt.size for cnt in skeleton_contours])
        biggest_contour = skeleton_contours[biggest_contour_index]

        return biggest_contour, True

    else:

        return np.ndarray(shape=(0, 1, 2), dtype=np.int32), False  # empty contour


def plot_radial_section(contour, board_center_coordinate, mid_cad_inner, mid_cad_outer):
    # debug plot
    transition_cads = get_cads(contour, board_center_coordinate)
    transition_cads.sort(key=lambda c: c.distance)
    triple_distance = mid_cad_inner.distance
    transition_distances = [transition_cad.distance for transition_cad in transition_cads]
    double_distance = mid_cad_outer.distance
    distances = np.array([triple_distance] + transition_distances + [double_distance])
    triple_angle = mid_cad_inner.angle
    transition_angles = [transition_cad.angle for transition_cad in transition_cads]
    double_angle = mid_cad_outer.angle
    angles = np.array([triple_angle] + transition_angles + [double_angle])
    new_coordinates_quantity = 100
    savgol_window_length = 31
    new_cads = get_interpolated_coordinates_with_smoothed_angles(
        angles,
        distances,
        board_center_coordinate,
        new_coordinates_quantity,
        savgol_window_length,
        mirror=False
    )
    new_angles = [new_cad.angle for new_cad in new_cads]
    new_distances = [new_cad.distance for new_cad in new_cads]
    plt.plot(distances, angles, label="radial section transition raw")
    plt.plot(new_distances, new_angles, label=f"savgol interp, window length: {savgol_window_length}")
    plt.xlabel("distances")
    plt.ylabel("angles")
    plt.legend()
    plt.show()


def middle_of_three_index(num1, num2, num3):
    # Calculate the sum of all three numbers
    total = num1 + num2 + num3

    # Find the minimum and maximum values
    min_num = min(num1, num2, num3)
    max_num = max(num1, num2, num3)

    # Calculate the middle value
    middle = total - min_num - max_num

    if num1 == middle:
        return 0
    elif num2 == middle:
        return 1
    elif num3 == middle:
        return 2
    else:
        return -1


def get_mask_quality(masked_img, lower_bound, upper_bound):
    """
    Returns 0 if the mask is okay,
    1 if too much percentage of the image is white
    and -1 if too less of the image is white
    """
    white_percentage = np.sum(masked_img == 255) / (masked_img.size / 2)
    if white_percentage < lower_bound:
        return -1
    elif white_percentage > upper_bound:
        return 1
    else:
        return 0


def get_test_images_for_contours(contours):
    test_images = [get_test_img_for_contour(cnt) for cnt in contours]
    return test_images


def get_board_rg_start(frame):

    (board_center_coordinate,
     bull_segment,
     bulls_eye_segment,
     contours_double,
     contours_triple,
     equalized_bgr,
     equalized_bgr_blurred,
     equalized_hsv_blurred,
     img_height,
     img_width) = get_triple_and_double_contours(frame)

    if len(contours_triple) != 20:
        triple_images = get_test_images_for_contours(contours_triple)
        raise AssertionError("Expected 20 triple contours, got " + str(len(contours_triple)))

    if len(contours_double) != 20:
        double_images = get_test_images_for_contours(contours_double)
        raise AssertionError("Expected 20 double contours, got " + str(len(contours_double)))

    equalized_bgr_contours = equalized_bgr.copy()
    cv.drawContours(equalized_bgr_contours, contours_triple, -1, (255, 255, 0), 1)
    cv.drawContours(equalized_bgr_contours, contours_double, -1, (0, 255, 255), 1)

    board_segments_triple = get_as_board_segments(board_center_coordinate, contours_triple, SegmentType.TRIPLE)
    board_segments_triple.sort(key=lambda x: x.center_cad.angle)
    board_segments_double = get_as_board_segments(board_center_coordinate, contours_double, SegmentType.DOUBLE)
    board_segments_double.sort(key=lambda x: x.center_cad.angle)

    # smooth rings, bulls_eye, bull, inner_triple, outer_triple, inner_double, outer_double
    smoothed_bulls_eye_contour = get_smoothed_bull_or_bulls_eye_contour(img_width, img_height, bulls_eye_segment)
    board_center_coordinate = get_center_of_contour(smoothed_bulls_eye_contour)
    bulls_eye_segment = get_as_board_segment(board_center_coordinate, smoothed_bulls_eye_contour, SegmentType.BULLS_EYE)

    smoothed_bull_contour = get_smoothed_bull_or_bulls_eye_contour(img_width, img_height, bull_segment)
    bull_segment = get_as_board_segment(board_center_coordinate, smoothed_bull_contour, SegmentType.BULL)

    smoothed_inner_triple_ring_cads, smoothed_outer_triple_ring_cads = get_smoothed_ring_cads(contours_triple, img_width, img_height, board_center_coordinate, equalized_bgr)
    smoothed_inner_triple_ring_contour = get_cads_as_contour(smoothed_inner_triple_ring_cads)
    smoothed_outer_triple_ring_contour = get_cads_as_contour(smoothed_outer_triple_ring_cads)
    smoothed_inner_double_ring_cads, smoothed_outer_double_ring_cads = get_smoothed_ring_cads(contours_double, img_width, img_height, board_center_coordinate, equalized_bgr)
    smoothed_inner_double_ring_contour = get_cads_as_contour(smoothed_inner_double_ring_cads)
    smoothed_outer_double_ring_contour = get_cads_as_contour(smoothed_outer_double_ring_cads)
    smoothed_ring_contours = [
        smoothed_bulls_eye_contour,
        smoothed_bull_contour,
        smoothed_inner_triple_ring_contour,
        smoothed_outer_triple_ring_contour,
        smoothed_inner_double_ring_contour,
        smoothed_outer_double_ring_contour
    ]

    # debug visualization
    equalized_bgr_rings_smoothed = equalized_bgr_blurred.copy()
    cv.drawContours(equalized_bgr_rings_smoothed, contours_triple, -1, (255, 255, 255), -1)
    cv.drawContours(equalized_bgr_rings_smoothed, contours_double, -1, (255, 255, 255), -1)
    cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_bulls_eye_contour], -1, (0, 255, 255), 2)
    cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_bull_contour], -1, (255, 255, 0), 2)
    cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_inner_triple_ring_contour], -1, (255, 0, 255), 2)
    cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_outer_triple_ring_contour], -1, (255, 255, 0), 2)
    cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_inner_double_ring_contour], -1, (255, 0, 255), 2)
    cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_outer_double_ring_contour], -1, (255, 255, 0), 2)

    # get the numbers for the double segments, use the bulls_eye_cp
    # board_segments_double = read_numbers_and_add_numbers_to_double_segments(
    #     equalized_bgr_blurred,
    #     img_width,
    #     img_height,
    #     board_center_coordinate,
    #     board_segments_double
    # )

    # get triple-double pairs of radial sections
    section_pairs: List[Tuple[BoardSegment, BoardSegment]]
    if abs(board_segments_triple[0].center_cad.angle - board_segments_double[0].center_cad.angle) < 1:
        section_pairs = [(cadc_t, cadcn_d) for cadc_t, cadcn_d in zip(board_segments_triple, board_segments_double)]
    else:
        if board_segments_triple[0].center_cad.angle < board_segments_double[0].center_cad.angle:
            board_segments_double = [board_segments_double[-1]] + board_segments_double[:-1]
        else:
            board_segments_triple = [board_segments_triple[-1]] + board_segments_triple[:-1]
        section_pairs = [(cadc_t, cadc_d) for cadc_t, cadc_d in zip(board_segments_triple, board_segments_double)]

    # Pair the triple-double pairs so the struct holds two adjacent triples and the corresponding two adjacent doubles
    section_triple_double_pairs = [(section_pairs[i], (section_pairs[(i + 1) % len(section_pairs)])) for i in range(len(section_pairs))]
    all_radial_section_transition_cads = []
    inner_radial_transitions_not_found_quantity = 0
    outer_radial_transitions_not_found_quantity = 0
    radial_sections_ring_intersections = []
    equalized_bgr_mid_coordinates = equalized_bgr.copy()
    for section_pair in section_triple_double_pairs:
        segment_triple_low_angle = section_pair[0][0]
        segment_double_low_angle = section_pair[0][1]
        segment_triple_high_angle = section_pair[1][0]
        segment_double_high_angle = section_pair[1][1]

        transition_cads_quantity = 5
        mid_cad_triple = get_mid_cad(segment_triple_low_angle, segment_triple_high_angle, board_center_coordinate)
        triple_distance_range = np.linspace(mid_cad_triple.distance * 0.98, mid_cad_triple.distance * 1.02, transition_cads_quantity)
        triple_transition_cads = get_endpoint_range(board_center_coordinate, [mid_cad_triple.angle] * transition_cads_quantity, triple_distance_range)

        mid_cad_double = get_mid_cad(segment_double_low_angle, segment_double_high_angle, board_center_coordinate)
        double_distance_range = np.linspace(mid_cad_double.distance * 0.98, mid_cad_double.distance * 1.02, transition_cads_quantity)
        double_transition_cads = get_endpoint_range(board_center_coordinate, [mid_cad_double.angle] * transition_cads_quantity, double_distance_range)

        box_corner_low_angle_low_distance_outer = get_line_extension_end_point(
            segment_triple_low_angle.center_cad.coordinate,
            segment_double_low_angle.center_cad.coordinate,
            0.2
        )

        box_corner_low_angle_high_distance_outer = get_line_extension_end_point(
            segment_double_low_angle.center_cad.coordinate,
            segment_triple_low_angle.center_cad.coordinate,
            0.2
        )

        box_corner_high_angle_high_distance_outer = get_line_extension_end_point(
            segment_double_high_angle.center_cad.coordinate,
            segment_triple_high_angle.center_cad.coordinate,
            0.2
        )

        box_corner_high_angle_low_distance_outer = get_line_extension_end_point(
            segment_triple_high_angle.center_cad.coordinate,
            segment_double_high_angle.center_cad.coordinate,
            0.2
        )
        box_coordinates_outer = [
            box_corner_low_angle_low_distance_outer,
            box_corner_low_angle_high_distance_outer,
            box_corner_high_angle_high_distance_outer,
            box_corner_high_angle_low_distance_outer
        ]

        outer_transition_contour, outer_transition_could_be_found = get_transition_contour_from_box(
            equalized_hsv_blurred,
            equalized_bgr_blurred,
            box_coordinates_outer
        )
        if not outer_transition_could_be_found:
            outer_radial_transitions_not_found_quantity += 1

        box_corner_low_angle_low_distance_inner = get_line_extension_end_point(
            board_center_coordinate,
            segment_triple_low_angle.center_cad.coordinate,
            0.3
        )

        box_corner_low_angle_high_distance_inner = get_line_extension_end_point(
            segment_triple_low_angle.center_cad.coordinate,
            board_center_coordinate,
            0.2
        )

        box_corner_high_angle_high_distance_inner = get_line_extension_end_point(
            segment_triple_high_angle.center_cad.coordinate,
            board_center_coordinate,
            0.2
        )

        box_corner_high_angle_low_distance_inner = get_line_extension_end_point(
            board_center_coordinate,
            segment_triple_high_angle.center_cad.coordinate,
            0.3
        )

        box_coordinates_inner = [
            box_corner_low_angle_low_distance_inner,
            box_corner_low_angle_high_distance_inner,
            box_corner_high_angle_high_distance_inner,
            box_corner_high_angle_low_distance_inner
        ]

        inner_transition_contour, inner_transition_could_be_found = get_transition_contour_from_box(
            equalized_hsv_blurred,
            equalized_bgr_blurred,
            box_coordinates_inner
        )
        if not inner_transition_could_be_found:
            inner_radial_transitions_not_found_quantity += 1

        # get two more coordinates to the radial section transition:
        # - the bull segment coordinate with the adjacent angle as the current contour
        # - a coordinate behind the double mid-coordinate as seen from board center coordinate
        # that way can close contour gaps when the contour is drawn later
        inner_transition_cads = get_cads(inner_transition_contour, board_center_coordinate)
        inner_transition_cads.sort(key=lambda x: x.distance)
        bull_segment_angles = [cad.angle for cad in bull_segment.contour_cads]
        if inner_transition_could_be_found:
            angle_to_compare = triple_transition_cads[0].angle

        else:
            angle_to_compare = mid_cad_triple.angle  # not the best alternative but better than nothing
        angle_diffs = [get_angle_diff(bs_angle, angle_to_compare) for bs_angle in bull_segment_angles]
        adjacent_board_segment_cad_index = np.argmin(angle_diffs)
        adjacent_bull_segment_cad = bull_segment.contour_cads[adjacent_board_segment_cad_index]

        behind_double_mid_coordinate = get_line_extension_end_point(board_center_coordinate, mid_cad_double.coordinate, 1.07)
        behind_double_mid_cad = get_cad(behind_double_mid_coordinate, board_center_coordinate)
        outer_transition_cads = get_cads(outer_transition_contour, board_center_coordinate)

        # glue the transition cads together and add the triple and double mid-coordinates for a complete radial
        # section transition
        radial_section_transition_cads = [adjacent_bull_segment_cad] + inner_transition_cads + triple_transition_cads + outer_transition_cads + double_transition_cads + [behind_double_mid_cad]
        all_radial_section_transition_cads.append(radial_section_transition_cads)

        # save all relevant radial-section-transition x ring-intersections for image warping
        cv.circle(equalized_bgr_mid_coordinates, (adjacent_bull_segment_cad.coordinate.x, adjacent_bull_segment_cad.coordinate.y), 2, (0, 255, 0), -1)
        triple_inner_transition = get_cad_with_adjacent_angle(smoothed_inner_triple_ring_cads, mid_cad_triple.angle)
        cv.circle(equalized_bgr_mid_coordinates, (triple_inner_transition.coordinate.x, triple_inner_transition.coordinate.y), 3, (0, 255, 0), -1)
        triple_outer_transition = get_cad_with_adjacent_angle(smoothed_outer_triple_ring_cads, mid_cad_triple.angle)
        cv.circle(equalized_bgr_mid_coordinates, (triple_outer_transition.coordinate.x, triple_outer_transition.coordinate.y), 3, (0, 255, 0), -1)
        double_inner_transition = get_cad_with_adjacent_angle(smoothed_inner_double_ring_cads, mid_cad_double.angle)
        cv.circle(equalized_bgr_mid_coordinates, (double_inner_transition.coordinate.x, double_inner_transition.coordinate.y), 3, (0, 255, 0), -1)
        double_outer_transition = get_cad_with_adjacent_angle(smoothed_outer_double_ring_cads, mid_cad_double.angle)
        cv.circle(equalized_bgr_mid_coordinates, (double_outer_transition.coordinate.x, double_outer_transition.coordinate.y), 3, (0, 255, 0), -1)
        radial_section_ring_intersections = RadialSectionRingIntersections(
            cads=[
                # adjacent_bull_segment_cad,
                triple_inner_transition,
                triple_outer_transition,
                double_inner_transition,
                double_outer_transition,
            ],
            number=segment_double_low_angle.number
        )
        radial_sections_ring_intersections.append(radial_section_ring_intersections)

    # warp the image 20 times with variations of the radial-section-transition x ring-intersections
    # then get the image that shows the correct numbers on the correct positions in warped image
    corner_variations = []
    for _ in range(20):
        first_radial_intersection = radial_sections_ring_intersections.pop(0)
        radial_sections_ring_intersections.append(first_radial_intersection)
        all_radial_corners = [CoordinateAngleDistance(board_center_coordinate, np.float32(0), np.float32(0))]
        for radial_corners in radial_sections_ring_intersections:
            corners = radial_corners.get_by_distance()
            all_radial_corners.extend(corners)
        corners_np = np.asarray([(cad.coordinate.x, cad.coordinate.y) for cad in all_radial_corners])
        corner_variations.append(corners_np)

    obj_points, img_dims = get_real_world_coordinates()
    # we have just this one image of the board, so we have to roll with that
    # calibrateCamera() however needs a list of obj_points and img_points for calibration
    obj_points_list = [obj_points]
    undistorted_images = []
    equalized_grey = cv.cvtColor(equalized_bgr, cv.COLOR_BGR2GRAY)
    for img_points in corner_variations:
        img_points = np.asarray(img_points, dtype=np.float32)
        img_points_list = [img_points]
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points_list, img_points_list, equalized_grey.shape[::-1], None, None)
        # mtx: Camera Matrix
        # dist: distortion coefficient
        # Position of camera in the real world
        # rvecs: rotation vectors
        # tvecs: translation vectors
        h, w = equalized_bgr.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # undistort
        dst_undistort = cv.undistort(equalized_bgr, mtx, dist, None, newcameramtx)
        undistorted_images.append(dst_undistort)
        # normally this would be a for loop, but we have only one image
        # there is no mean_error ether, just a solo error
        img_points_2, _ = cv.projectPoints(obj_points, rvecs[0], tvecs[0], mtx, dist)
        error = cv.norm(img_points, img_points_2, cv.NORM_L2) / len(img_points_2)
        print("error for img variation: {}".format(error / len(obj_points)))

    undistorted_img = undistorted_images[-1]
    undistorted_img = cv.cvtColor(undistorted_img, cv.COLOR_BGR2RGB)
    plt.imshow(undistorted_img)
    plt.show()

    equalized_bgr_corners = equalized_bgr.copy()
    for radial_corners in radial_sections_ring_intersections:
        for radial_corner in radial_corners.get_by_distance():
            cv.circle(equalized_bgr_corners, (radial_corner.coordinate.x, radial_corner.coordinate.y), 1, (0, 255, 0), -1)

    # from preset_circled_board_coordinates2 import get_preset_board_coordinates
    # preset_corners, img_dims = get_preset_board_coordinates()
    # image_warps = []
    # for board_corners_live in corner_variations:
    #     transformed_img = transform_img(equalized_bgr, board_corners_live, preset_corners, img_dims)
    #     image_warps.append((transformed_img, board_corners_live))

    # image_warps_segmented = []
    # for warped_img, board_corners in [image_warps[0]]:
    #     homography = get_homography(preset_corners, board_corners)
    #     score_segments, score_segment_img = get_preset_board_segments(homography, warped_img)
    #     image_warps_segmented.append(score_segment_img)

    if inner_radial_transitions_not_found_quantity:
        print(f"{inner_radial_transitions_not_found_quantity} inner radial transitions not found.")
    if outer_radial_transitions_not_found_quantity:
        print(f"{outer_radial_transitions_not_found_quantity} outer radial transitions not found.")

    t_radial_transitions_img = equalized_bgr_blurred.copy()
    for radial_section_transition_cads in all_radial_section_transition_cads:
        radial_section_transition_contour = np.asarray([(cad.coordinate.x, cad.coordinate.y) for cad in radial_section_transition_cads])
        cv.polylines(t_radial_transitions_img, [radial_section_transition_contour], isClosed=False, color=(255, 255, 255), thickness=2)

    radial_section_transitions_smoothed = smooth_transition_point_groups(board_center_coordinate, all_radial_section_transition_cads)

    t_radial_transitions_img_smoothed = equalized_bgr_blurred.copy()
    for radial_section_transition_cads in radial_section_transitions_smoothed:
        radial_section_transition_contour = np.asarray(
            [(cad.coordinate.x, cad.coordinate.y) for cad in radial_section_transition_cads])
        cv.polylines(t_radial_transitions_img_smoothed, [radial_section_transition_contour], isClosed=False,
                     color=(255, 255, 255), thickness=2)

    radial_section_transitions_smoothed = [np.asarray([(cad.coordinate.x, cad.coordinate.y) for cad in rst]) for rst in radial_section_transitions_smoothed]

    # get new data structure for each double contour, CoordinateAngleDistanceContourNumber, cadcn
    # save the all segment center coordinates, segment type and their number in a structure
    c_st_n_s: List[CoordinateSegmentTypeNumber] = []
    c_st_n_s.append(CoordinateSegmentTypeNumber(board_center_coordinate, SegmentType.BULLS_EYE, None))
    c_st_n_s.append(CoordinateSegmentTypeNumber(bull_segment.center_cad.coordinate, SegmentType.BULL, None))
    for td_pair in section_pairs:
        triple_segment, double_segment = td_pair
        expected_inner_segment_cp = get_line_extension_end_point(board_center_coordinate, triple_segment.center_cad.coordinate, 0.65)
        cst_inner = CoordinateSegmentTypeNumber(expected_inner_segment_cp, SegmentType.INNER, double_segment.number)
        cst_triple = CoordinateSegmentTypeNumber(triple_segment.center_cad.coordinate, SegmentType.TRIPLE, double_segment.number)
        expected_outer_segment_cp = get_mid_coordinate(triple_segment.center_cad.coordinate, double_segment.center_cad.coordinate)
        cst_outer = CoordinateSegmentTypeNumber(expected_outer_segment_cp, SegmentType.OUTER, double_segment.number)
        cst_double = CoordinateSegmentTypeNumber(double_segment.center_cad.coordinate, SegmentType.DOUBLE, double_segment.number)
        c_st_n_s.extend([cst_inner, cst_triple, cst_outer, cst_double])

    # draw the rings and side_coordinates onto a blank image
    score_segment_img = np.zeros(frame.shape[:2], dtype="uint8")
    cv.drawContours(score_segment_img, smoothed_ring_contours, contourIdx=-1, color=(255, 255, 255), thickness=2)
    for rsts in radial_section_transitions_smoothed:
        score_segment_img = cv.polylines(score_segment_img, [rsts], isClosed=False, color=(255, 255, 255), thickness=2)
    # invert the image
    score_segment_img_inverted = cv.bitwise_not(score_segment_img)
    # get contours from the image
    contours, _ = cv.findContours(score_segment_img_inverted, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contour_distances_to_bulls_eye = [
        get_distance(get_center_of_contour(cnt), bulls_eye_segment.center_cad.coordinate)
        for cnt in contours
    ]
    cnt_cnt_dists = list(zip(contours, contour_distances_to_bulls_eye))
    cnt_cnt_dists.sort(key=lambda x: x[1])
    bulls_eye_bull_candidates = cnt_cnt_dists[:3]
    bulls_eye_bull_candidate_cnt_areas = [cv.contourArea(cnt[0]) for cnt in bulls_eye_bull_candidates]
    false_bulls_eye_candidate_index = middle_of_three_index(bulls_eye_bull_candidate_cnt_areas[0], bulls_eye_bull_candidate_cnt_areas[1], bulls_eye_bull_candidate_cnt_areas[2])
    cnt_cnt_dists.pop(false_bulls_eye_candidate_index)
    cnt_cnt_areas = [(cnt[0], cv.contourArea(cnt[0])) for cnt in cnt_cnt_dists]
    cnt_cnt_areas.sort(key=lambda x: x[1])
    cnt_cnt_areas.pop(-1)  # everything else around the dartboard
    cnt_cnt_areas.pop(-1)  # convex hull of the drawn dartboard

    contours_filtered = [cnt_cnt_area[0] for cnt_cnt_area in cnt_cnt_areas]

    if len(contours_filtered) != 82:
        raise AssertionError(f"Expected 82 drawn contours, got {len(contours_filtered)}")

    # get center_point for each contour
    coordinates_contours = [CoordinateContour(get_center_of_contour(cnt), cnt) for cnt in contours_filtered]
    # for each saved segment_center_point search for the nearest contour_center_point
    score_segments: List[ScoreSegment] = []
    for cstn in c_st_n_s:
        score_segment_distances = [get_distance(cstn.coordinate, cc.coordinate) for cc in coordinates_contours]
        min_distance_segment_index = np.argmin(score_segment_distances)
        min_distance_segment = coordinates_contours[min_distance_segment_index]
        score_segments.append(ScoreSegment(min_distance_segment.contour, cstn.segment_type, cstn.number))
        coordinates_contours.pop(min_distance_segment_index)  # reduces "complexity" to 3240 distance calculations

    # because bull and bullseye have nearly the same center coordinate, they may have the wrong segment type
    score_segment_bulls_eye = [sc for sc in score_segments if sc.segment_type == SegmentType.BULLS_EYE][0]
    score_segment_bull = [sc for sc in score_segments if sc.segment_type == SegmentType.BULL][0]
    if cv.contourArea(score_segment_bulls_eye.contour) > cv.contourArea(score_segment_bull.contour):
        score_segment_bulls_eye.segment_type = SegmentType.BULL
        score_segment_bull.segment_type = SegmentType.BULLS_EYE

    if len(score_segments) != 82:
        raise AssertionError(f"Expected 82 Score-segments, was {len(score_segments)}")

    return True, score_segments, score_segment_img, radial_section_ring_intersections


def get_triple_and_double_contours(frame):
    img_height = frame.shape[0]
    img_width = frame.shape[1]
    ycrcb_img = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb_img)
    y_equalized = cv.equalizeHist(y)
    ycrcb = cv.merge((y_equalized, cr, cb))
    equalized_bgr = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)
    equalized_hsv = cv.cvtColor(equalized_bgr, cv.COLOR_BGR2HSV)
    blur = cv.blur(frame, (5, 5))
    ycrcb_img = cv.cvtColor(blur, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb_img)
    y_equalized = cv.equalizeHist(y)
    ycrcb = cv.merge((y_equalized, cr, cb))
    equalized_bgr_blurred = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)
    equalized_hsv_blurred = cv.cvtColor(equalized_bgr_blurred, cv.COLOR_BGR2HSV)
    hsv_mask_red_local = copy(hsv_mask_red)
    hsv_mask_green_local = copy(hsv_mask_green)
    masked_img_red = get_masked_img_by_hsv_values(equalized_hsv_blurred, hsv_mask_red_local)
    masked_img_green = get_masked_img_by_hsv_values(equalized_hsv_blurred, hsv_mask_green_local)
    # get a parameter to evaluate the mask quality
    red_lower_bound = 0.06
    red_upper_bound = 0.085
    green_lower_bound = 0.05
    green_upper_bound = 0.063
    saturation_step = 2
    mask_quality_red = get_mask_quality(masked_img_red, red_lower_bound, red_upper_bound)
    while mask_quality_red != 0:
        if mask_quality_red < 0:
            hsv_mask_red_local.sat_lower -= saturation_step
        else:
            hsv_mask_red_local.sat_lower += saturation_step
        masked_img_red = get_masked_img_by_hsv_values(equalized_hsv_blurred, hsv_mask_red_local)
        mask_quality_red = get_mask_quality(masked_img_red, red_lower_bound, red_upper_bound)
    mask_quality_green = get_mask_quality(masked_img_green, green_lower_bound, green_upper_bound)
    while mask_quality_green != 0:
        if mask_quality_green < 0:
            hsv_mask_green_local.sat_lower -= saturation_step
        else:
            hsv_mask_green_local.sat_lower += saturation_step
        masked_img_green = get_masked_img_by_hsv_values(equalized_hsv_blurred, hsv_mask_green_local)
        mask_quality_green = get_mask_quality(masked_img_green, green_lower_bound, green_upper_bound)
    # masked_img_red_eroded = cv.erode(masked_img_red, np.ones((3, 3), np.uint8))
    # masked_img_green_eroded = cv.erode(masked_img_green, np.ones((3, 3), np.uint8))
    masked_img_red_filtered = get_filtered_triple_double_mask(masked_img_red)
    masked_img_green_filtered = get_filtered_triple_double_mask(masked_img_green)
    contour_bulls_eye, contour_bull = get_bull_and_bulls_eye_contour(masked_img_green)
    board_center_coordinate = get_center_of_contour(contour_bull)  # is more precise than bulls-eye
    bull_segment = get_as_board_segment(board_center_coordinate, contour_bull, SegmentType.BULL)
    bulls_eye_segment = get_as_board_segment(board_center_coordinate, contour_bulls_eye, SegmentType.BULLS_EYE)
    red_and_green_contours_img = cv.bitwise_or(masked_img_red_filtered, masked_img_green_filtered)
    dilated_rg_contours_img = cv.dilate(red_and_green_contours_img, np.ones((20, 20), np.uint8))
    triples_convex_hull_mask = get_third_biggest_convex_hull(dilated_rg_contours_img)
    triples_convex_hull_mask_inv = cv.bitwise_not(triples_convex_hull_mask)
    # go from board center coordinate to each contour in
    img_triples_red = cv.bitwise_and(masked_img_red_filtered, triples_convex_hull_mask)
    img_triples_green = cv.bitwise_and(masked_img_green_filtered, triples_convex_hull_mask)
    # don't approximate contours, to not get too much variation in the number of points per contour
    red_double_contours, red_triple_contours = match_double_contours(
        board_center_coordinate,
        img_triples_red,
        masked_img_red_filtered,
        triples_convex_hull_mask_inv
    )
    green_double_contours, green_triple_contours = match_double_contours(
        board_center_coordinate,
        img_triples_green,
        masked_img_green_filtered,
        triples_convex_hull_mask_inv
    )
    contours_triple = red_triple_contours + green_triple_contours
    contours_double = red_double_contours + green_double_contours

    img_double_triple_mid = np.zeros_like(frame)
    cv.drawContours(img_double_triple_mid, contours_triple, -1, (0, 255, 0), -1)
    cv.drawContours(img_double_triple_mid, contours_double, -1, (255, 255, 0), -1)
    for cnt in contours_triple + contours_double:
        cnt_center = get_center_of_contour(cnt)
        cv.circle(img_double_triple_mid, (cnt_center.x, cnt_center.y), 2, (255, 255, 255), -1)
    return board_center_coordinate, bull_segment, bulls_eye_segment, contours_double, contours_triple, equalized_bgr, equalized_bgr_blurred, equalized_hsv_blurred, img_height, img_width


def match_double_contours(board_center_coordinate, img_triples, masked_img_filtered, triples_convex_hull_mask_inv):
    triple_contours, _ = cv.findContours(img_triples, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    triple_cadcs = get_cadcs_for_contours(board_center_coordinate, triple_contours)
    img_double_candidates = cv.bitwise_and(masked_img_filtered, triples_convex_hull_mask_inv)
    double_candidates, _ = cv.findContours(img_double_candidates, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    double_cadcs = get_cadcs_for_contours(board_center_coordinate, double_candidates)
    double_contours = []
    for triple_cadc in triple_cadcs:
        matching_double_cadcs = [cadc for cadc in double_cadcs if
                                 is_angle_in_range(cadc.angle, (triple_cadc.angle - 0.05) % (2 * np.pi),
                                                   (triple_cadc.angle + 0.05) % (2 * np.pi))]
        matching_double_cadc = min(matching_double_cadcs, key=lambda x: x.distance)
        double_cadcs.remove(matching_double_cadc)
        double_contours.append(matching_double_cadc.contour)
    return double_contours, list(triple_contours)


def get_third_biggest_convex_hull(dilate_test):
    contours, _ = cv.findContours(dilate_test, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    third_biggest_contour = contours[2]
    hull = cv.convexHull(third_biggest_contour)

    convex_hull_mask = np.zeros(dilate_test.shape, np.uint8)
    cv.drawContours(convex_hull_mask, [hull], -1, (255, 255, 255), -1)

    return convex_hull_mask


def get_bull_and_bulls_eye_contour(masked_img_green):
    contours, _ = cv.findContours(masked_img_green, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    bull_or_bulls_eye_contours = []
    for cnt in contours:
        is_round = filter_max_elongation(cnt, 1.52)
        is_solid = filter_contour_solidity(cnt, 0.9)
        is_big_enough = filter_min_size_contour(cnt, 300)
        is_not_too_big = filter_max_size_contour(cnt, 8_000)
        if is_round and is_solid and is_big_enough and is_not_too_big:
            bull_or_bulls_eye_contours.append(cnt)

    if len(bull_or_bulls_eye_contours) != 2:
        contours_sizes = [(cnt, cv.contourArea(cnt)) for cnt in contours]
        contours_sizes.sort(key=lambda x: x[1], reverse=True)
        top_30_contours = [contour_size[0] for contour_size in contours_sizes[:30]]
        images_for_top_30 = [get_test_img_for_contour(cnt) for cnt in top_30_contours]
        raise LookupError("Expected two contours (one bull contour and one bulls-eye contour), found " + str(len(bull_or_bulls_eye_contours)))
    bulls_eye_contour = min(bull_or_bulls_eye_contours, key=cv.contourArea)
    bull_contour = max(bull_or_bulls_eye_contours, key=cv.contourArea)

    return bulls_eye_contour, bull_contour


def get_filtered_triple_double_mask(masked_img) -> np.ndarray:
    # delete everything but triple and double contours
    contours, _ = cv.findContours(masked_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    wanted_contours = []
    for cnt in contours:
        is_solid = filter_contour_solidity(cnt, 0.6)
        is_elongated = filter_min_elongation(cnt, 2)
        is_big_enough = filter_min_size_contour(cnt, 300)
        hull = cv.convexHull(cnt, returnPoints=False)
        # check if the hull indices are homogenous, if not, the contour has self-intersections and can be discarded
        if not np.all(hull[:-1] <= hull[1:]) and not np.all(hull[:-1] >= hull[1:]):
            has_no_severe_defects = False
        else:
            defects = cv.convexityDefects(cnt, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    if d > 5000:
                        cv.circle(masked_img, far, 5, [80, 80, 80], -1)
        if is_solid and is_elongated and is_big_enough:
            wanted_contours.append(cnt)

    new_contour_img = np.zeros(masked_img.shape, np.uint8)
    cv.drawContours(new_contour_img, wanted_contours, -1, (255, 255, 255), -1)

    return new_contour_img