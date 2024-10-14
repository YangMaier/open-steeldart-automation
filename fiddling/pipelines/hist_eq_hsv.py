import copy
import random
import time
from typing import List, Tuple

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from data_structures.coordinates import Coordinate2d
from data_structures.coordinates_extended import CoordinateAngleDistanceContour, CoordinateAngleDistance, \
    CoordinateGroup, CoordinateSegmentTypeNumber, CoordinateContour, CoordinateAngleDistanceContourNumber
from data_structures.interpolation import AngleDistanceInterpolation, EdgeValues
from data_structures.score_segments import SegmentType, ScoreSegment
from image_masking import get_masked_img_by_hsv_values
from operations_on.angles import get_mid_angle
from operations_on.contours import get_center_of_contour, get_interpolations
from operations_on.coordinates import get_distance, get_mid_coordinate, get_interpolation, get_points_as_nparray
from operations_on.coordinates_and_angles import get_angle, get_endpoint_range, get_cads, get_edge_values
from data_structures.hsv_mask_presets import HSVMaskRedEq, HSVMaskGreenEq, HSVMaskWhiteEq, HSVMaskBlackEq
from data_structures.segment_template import ExpectedTemplatesRed, ExpectedTemplatesGreen
from contour_template_matching import get_relevant_segments
from operations_on.ellipses import get_fit_ellipse
from operations_on.letters import read_numbers_and_add_numbers_to_double_segments


def nothing(x):
    pass


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (
            graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count']
    )


def get_smoothed_bull_or_bulls_eye_contour(img_width, img_height, bulls_eye_cp, contour) -> np.ndarray:
    cads = get_cads(contour, bulls_eye_cp)
    angles, smoothed_distances = get_interpolation(cads)
    smoothed_points = get_endpoint_range(bulls_eye_cp, angles, smoothed_distances, safe_angle_and_distance=False)
    smoothed_points = get_points_as_nparray(smoothed_points)
    ellipse_points = get_fit_ellipse(smoothed_points, img_width, img_height)
    ellipse_points = get_cads(ellipse_points, bulls_eye_cp)
    smoothed_ellipse_angles, smoothed_ellipse_distances = get_interpolation(ellipse_points)
    # mean_distances = np.mean([smoothed_distances, smoothed_ellipse_distances], axis=0)

    smoothed_contour = get_endpoint_range(bulls_eye_cp, smoothed_ellipse_angles, smoothed_ellipse_distances, safe_angle_and_distance=False)
    smoothed_contour = np.asarray([(c.x, c.y) for c in smoothed_contour])

    return smoothed_contour


def add_bull_connection_coordinates(smoothed_bull_contour, radial_section_transitions_smoothed, bulls_eye_cp):
    smoothed_bull_cads = get_cads(smoothed_bull_contour, bulls_eye_cp)
    for list_of_cads in radial_section_transitions_smoothed:
        nearest_coordinate_to_bull = list_of_cads[0].coordinate
        bull_distances_to_radial_section = [get_distance(nearest_coordinate_to_bull, b_c.coordinate) for b_c in smoothed_bull_cads]
        next_bull_coordinate_index = np.argmin(bull_distances_to_radial_section)
        next_bull_coordinate = smoothed_bull_cads[next_bull_coordinate_index]
        list_of_cads.insert(0, next_bull_coordinate)

    return radial_section_transitions_smoothed


def threshold_sliders():
    cam_id = 2
    cam = cv.VideoCapture(cam_id)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    cv.namedWindow('sliders')
    cv.createTrackbar('cnt dist', 'sliders', 0, 100, nothing)
    cv.createTrackbar('elon min', 'sliders', 0, 10, nothing)
    cv.createTrackbar('elon max', 'sliders', 10, 30, nothing)
    cv.createTrackbar('solid min', 'sliders', 0, 20, nothing)
    cv.createTrackbar('area min', 'sliders', 0, 10, nothing)
    cv.createTrackbar('area max', 'sliders', 5, 30, nothing)

    cv.setTrackbarPos('cnt dist', 'sliders', 12)
    cv.setTrackbarPos('elon min', 'sliders', 8)
    cv.setTrackbarPos('elon max', 'sliders', 12)
    cv.setTrackbarPos('solid min', 'sliders', 9)
    cv.setTrackbarPos('area min', 'sliders', 5)
    cv.setTrackbarPos('area max', 'sliders', 17)

    while True:
        frame_time = time.time_ns()
        img_height = frame.shape[0]
        img_width = frame.shape[1]

        blur = cv.blur(frame, (5, 5))

        scale_cnt_dist = cv.getTrackbarPos('cnt dist', 'sliders') / 100
        scale_elongation_min = cv.getTrackbarPos('elon min', 'sliders') / 10
        scale_elongation_max = cv.getTrackbarPos('elon max', 'sliders') / 10
        scale_solidity_min = cv.getTrackbarPos('solid min', 'sliders') / 10
        scale_area_min = cv.getTrackbarPos('area min', 'sliders') / 10
        scale_area_max = cv.getTrackbarPos('area max', 'sliders') / 10

        ycrcb_img = cv.cvtColor(blur, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(ycrcb_img)
        y_equalized = cv.equalizeHist(y)
        ycrcb = cv.merge((y_equalized, cr, cb))
        equalized_bgr = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

        hsv = cv.cvtColor(equalized_bgr, cv.COLOR_BGR2HSV)
        red_hsv_mask = HSVMaskRedEq()
        green_hsv_mask = HSVMaskGreenEq()


        masked_img_red = get_masked_img_by_hsv_values(hsv, red_hsv_mask)
        masked_img_green = get_masked_img_by_hsv_values(hsv, green_hsv_mask)

        cv.imshow("mask red", masked_img_red)
        cv.imshow("mask green", masked_img_green)

        segments_red, img_not_matched_red = get_relevant_segments(
            masked_img_red,
            ExpectedTemplatesRed(),
            scale_elongation_min,
            scale_elongation_max,
            scale_solidity_min,
            scale_area_min,
            scale_area_max,
            scale_cnt_dist,
            img_width,
            img_height,
            "red"
        )
        segments_green, img_not_matched_green = get_relevant_segments(
            masked_img_green,
            ExpectedTemplatesGreen(),
            scale_elongation_min,
            scale_elongation_max,
            scale_solidity_min,
            scale_area_min,
            scale_area_max,
            scale_cnt_dist,
            img_width,
            img_height,
            "green"
        )

        contours_bulls_eye = segments_red.bulls_eye_contours
        # green inner bull contour is used as fallback if the bullseye contour was not found in red mask
        if len(contours_bulls_eye) == 0:
            contours_bulls_eye = segments_green.bulls_eye_contours

        contours_bull = segments_green.bull_contours
        contours_triple = segments_red.triple_contours + segments_green.triple_contours
        contours_double = segments_red.double_contours + segments_green.double_contours

        # triple and double contours may be sorted in the wrong category, thats why the overall quantity should be 40
        # instead of checking len(triples) == 20 and len(doubles) == 20
        if len(contours_bulls_eye) == 1 and len(contours_bull) == 1 and (len(contours_triple) + len(contours_double)) == 40:
            bulls_eye_cp = get_center_of_contour(contours_bulls_eye[0])
            bull_segment = contours_bull[0]
            contours_triple, contours_double = correct_triple_and_double_contour_categories(bulls_eye_cp, contours_double, contours_triple)

            # find the radial section transition points of inner and outer segments
            convex_hull_bull = cv.convexHull(contours_bull[0])
            all_triple_points = np.concatenate(contours_triple, axis=0)
            convex_hull_triple_segments = cv.convexHull(all_triple_points)
            all_double_points = np.concatenate(contours_double, axis=0)
            convex_hull_double_segments = cv.convexHull(all_double_points)

            white_black_transition_cads_inner: List[CoordinateAngleDistance] = get_black_white_transition_points(
                equalized_bgr.copy(),
                convex_hull_bull,
                convex_hull_triple_segments,
                bulls_eye_cp
            )

            white_black_transition_cads_outer: List[CoordinateAngleDistance] = get_black_white_transition_points(
                equalized_bgr.copy(),
                convex_hull_triple_segments,
                convex_hull_double_segments,
                bulls_eye_cp
            )

            # debug visualization
            purple = (255, 0, 255)
            yellow = (0, 255, 255)
            equalized_bgr_raw_transitions = equalized_bgr.copy()
            for cad in white_black_transition_cads_inner + white_black_transition_cads_outer:
                cv.circle(equalized_bgr_raw_transitions, (cad.coordinate.x, cad.coordinate.y), 3, purple, 1)

            # group inner radial section transition points by angle to board center
            angle_diff_cutoff = 2 * np.pi / 60
            radial_section_inner_point_groups: List[CoordinateGroup] = group_cads_by_angle(
                white_black_transition_cads_inner,
                angle_diff_cutoff
            )
            radial_section_outer_point_groups: List[CoordinateGroup] = group_cads_by_angle(
                white_black_transition_cads_outer,
                angle_diff_cutoff
            )

            radial_section_inner_outer_coordinate_group_pairs = get_coordinate_group_pairs_by_angle(radial_section_inner_point_groups, radial_section_outer_point_groups)

            cadcs_triple = get_cadcs_for_contours(bulls_eye_cp, contours_triple)
            cadcs_triple.sort(key=lambda x: x.angle)
            cadcs_double = get_cadcs_for_contours(bulls_eye_cp, contours_double)
            cadcs_double.sort(key=lambda x: x.angle)

            radial_section_transitions: List[List[CoordinateAngleDistance]] = []
            for inner_outer_coordinate_group_pair in radial_section_inner_outer_coordinate_group_pairs:
                side_cads: List[CoordinateAngleDistance] = []
                inner_coordinate_group = inner_outer_coordinate_group_pair[0]
                side_cads.extend(inner_coordinate_group.cads)
                triple_with_next_smaller_angle, triple_with_next_bigger_angle = get_adjacent_ring_elements(cadcs_triple, inner_coordinate_group)
                triple_mid_distance = (triple_with_next_bigger_angle.distance + triple_with_next_smaller_angle.distance) / 2
                triple_mid_angle = get_angle_between_ring_elements(bulls_eye_cp, triple_with_next_smaller_angle, triple_with_next_bigger_angle)
                triple_mid_distance_lin = np.linspace(triple_mid_distance * 0.93, triple_mid_distance * 1.07, 10)
                triple_mid_coordinates = get_endpoint_range(bulls_eye_cp, [triple_mid_angle] * len(triple_mid_distance_lin), triple_mid_distance_lin)
                side_cads.extend(triple_mid_coordinates)

                outer_coordinate_group = inner_outer_coordinate_group_pair[1]
                side_cads.extend(outer_coordinate_group.cads)
                double_with_next_smaller_angle, double_with_next_bigger_angle = get_adjacent_ring_elements(cadcs_double, outer_coordinate_group)
                double_mid_distance = (double_with_next_bigger_angle.distance + double_with_next_smaller_angle.distance) / 2
                double_mid_angle = get_angle_between_ring_elements(bulls_eye_cp, double_with_next_smaller_angle, double_with_next_bigger_angle)
                double_mid_distance_lin = np.linspace(double_mid_distance * 0.8, double_mid_distance * 1.07, 10)
                double_mid_coordinates = get_endpoint_range(bulls_eye_cp, [double_mid_angle] * len(double_mid_distance_lin), double_mid_distance_lin)
                side_cads.extend(double_mid_coordinates)

                radial_section_transitions.append(side_cads)

            # debug visualization
            equalized_bgr_radial_sections = equalized_bgr.copy()
            for list_of_cads in radial_section_transitions:
                rand_int1 = random.randint(0, 255)
                rand_int2 = random.randint(0, 255)
                rand_int3 = random.randint(0, 255)
                for cad in list_of_cads:
                    cv.circle(equalized_bgr_radial_sections, (cad.coordinate.x, cad.coordinate.y), 3, (rand_int1, rand_int2, rand_int3), 1)

            radial_section_transitions_smoothed = smooth_transition_point_groups(bulls_eye_cp, radial_section_transitions)

            # smooth rings, bulls_eye, bull, inner_triple, outer_triple, inner_double, outer_double
            smoothed_bulls_eye_contour = get_smoothed_bull_or_bulls_eye_contour(img_width, img_height, bulls_eye_cp, contours_bulls_eye[0])
            smoothed_bull_contour = get_smoothed_bull_or_bulls_eye_contour(img_width, img_height, bulls_eye_cp, contours_bull[0])
            smoothed_inner_triple_ring_contour, smoothed_outer_triple_ring_contour = get_smoothed_ring_contours(contours_triple, img_width, img_height, bulls_eye_cp)
            smoothed_inner_double_ring_contour, smoothed_outer_double_ring_contour = get_smoothed_ring_contours(contours_double, img_width, img_height, bulls_eye_cp)
            smoothed_ring_contours = [
                smoothed_bulls_eye_contour,
                smoothed_bull_contour,
                smoothed_inner_triple_ring_contour,
                smoothed_outer_triple_ring_contour,
                smoothed_inner_double_ring_contour,
                smoothed_outer_double_ring_contour
            ]

            radial_section_transitions_smoothed = add_bull_connection_coordinates(smoothed_bull_contour, radial_section_transitions_smoothed, bulls_eye_cp)
            radial_section_transitions_smoothed = [np.asarray([(cad.coordinate.x, cad.coordinate.y) for cad in rst]) for rst in radial_section_transitions_smoothed]

            # debug visualization
            equalized_bgr_radial_sections_smoothed = equalized_bgr.copy()
            for contour in radial_section_transitions_smoothed:
                rand_int1 = random.randint(0, 255)
                rand_int2 = random.randint(0, 255)
                rand_int3 = random.randint(0, 255)
                cv.drawContours(equalized_bgr_radial_sections_smoothed, [contour], -1,
                                (rand_int1, rand_int2, rand_int3), 1)

            # debug visualization
            equalized_bgr_rings_smoothed = equalized_bgr.copy()
            cv.drawContours(equalized_bgr_rings_smoothed, contours_triple, -1, (255, 255, 255), -1)
            cv.drawContours(equalized_bgr_rings_smoothed, contours_double, -1, (255, 255, 255), -1)
            cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_bulls_eye_contour], -1, (0, 255, 255), 2)
            cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_bull_contour], -1, (255, 255, 0), 2)
            cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_inner_triple_ring_contour], -1, (255, 0, 255), 2)
            cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_outer_triple_ring_contour], -1, (255, 255, 0), 2)
            cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_inner_double_ring_contour], -1, (255, 0, 255), 2)
            cv.drawContours(equalized_bgr_rings_smoothed, [smoothed_outer_double_ring_contour], -1, (255, 255, 0), 2)

            # get the numbers for the double segments, use the bulls_eye_cp
            cadcns_double: List[CoordinateAngleDistanceContourNumber] = read_numbers_and_add_numbers_to_double_segments(equalized_bgr, img_width, img_height, bulls_eye_cp, cadcs_double)
            # get new data structure for each double contour, CoordinateAngleDistanceContourNumber, cadcn

            # build segments by smoothed side_point_groups and smoothed ring groups


            # get triple-double pairs
            triple_double_pairs: List[Tuple[CoordinateAngleDistanceContour, CoordinateAngleDistanceContourNumber]]
            if abs(cadcs_triple[0].angle - cadcns_double[0].angle) < 1:
                triple_double_pairs = [(cadc_t, cadcn_d) for cadc_t, cadcn_d in zip(cadcs_triple, cadcns_double)]
            else:
                if cadcs_triple[0].angle < cadcns_double[0].angle:
                    cadcns_double = [cadcns_double[-1]] + cadcns_double[:-1]
                else:
                    cadcs_triple = [cadcs_triple[-1]] + cadcs_triple[:-1]
                triple_double_pairs = [(cadc_t, cadc_d) for cadc_t, cadc_d in zip(cadcs_triple, cadcns_double)]

            # save the all segment center coordinates, segment type and their number in a structure
            cstns: List[CoordinateSegmentTypeNumber] = []
            cstns.append(CoordinateSegmentTypeNumber(bulls_eye_cp, SegmentType.BULLS_EYE, None))
            cstns.append(CoordinateSegmentTypeNumber(get_center_of_contour(bull_segment), SegmentType.BULL, None))
            for td_pair in triple_double_pairs:
                triple_cadc, cadcn_double = td_pair
                expected_inner_segment_cp = get_mid_coordinate(triple_cadc.coordinate, bulls_eye_cp)
                cst_inner = CoordinateSegmentTypeNumber(expected_inner_segment_cp, SegmentType.INNER, cadcn_double.number)
                cst_triple = CoordinateSegmentTypeNumber(triple_cadc.coordinate, SegmentType.TRIPLE, cadcn_double.number)
                expected_outer_segment_cp = get_mid_coordinate(triple_cadc.coordinate, cadcn_double.coordinate)
                cst_outer = CoordinateSegmentTypeNumber(expected_outer_segment_cp, SegmentType.OUTER, cadcn_double.number)
                cst_double = CoordinateSegmentTypeNumber(cadcn_double.coordinate, SegmentType.DOUBLE, cadcn_double.number)
                cstns.extend([cst_inner, cst_triple, cst_outer, cst_double])

            # draw the rings and side_coordinates onto a blank image
            score_segment_img = np.zeros(frame.shape[:2], dtype="uint8")
            cv.drawContours(score_segment_img, smoothed_ring_contours, contourIdx=-1, color=(255, 255, 255), thickness=2)
            for rsts in radial_section_transitions_smoothed:
                score_segment_img = cv.polylines(score_segment_img, [rsts], isClosed=False, color=(255, 255, 255), thickness=2)
            # invert the image
            score_segment_img_inverted = cv.bitwise_not(score_segment_img)
            # get contours from the image
            contours, _ = cv.findContours(score_segment_img_inverted, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            contours_filtered = []
            for cnt in contours:
                cnt_area = cv.contourArea(cnt)
                if 10 < cnt_area < img_width * img_height / 10:
                    contours_filtered.append(cnt)

            # get center_point for each contour
            coordinates_contours = [CoordinateContour(get_center_of_contour(cnt), cnt) for cnt in contours_filtered]
            # for each saved segment_center_point search for the nearest contour_center_point
            score_segments = []
            for cstn in cstns:
                score_segment_distances = [get_distance(cstn.coordinate, cc.coordinate) for cc in coordinates_contours]
                min_distance_segment_index = np.argmin(score_segment_distances)
                min_distance_segment = coordinates_contours[min_distance_segment_index]
                score_segments.append(ScoreSegment(min_distance_segment.contour, cstn.segment_type, cstn.number))
                coordinates_contours.pop(min_distance_segment_index)  # reduces complexity to 3240 distance calculations

            # make this a function and return only score_segments and if the image could be processed
            x = 0

        else:
            print("Not the expected number of bullseye-, bull-, triple- and double-segments found")

        img_bulls_eye_contours = np.zeros(frame.shape[:2], dtype="uint8")
        img_bull_contours = np.zeros(frame.shape[:2], dtype="uint8")
        # img_inner_contours = np.zeros(frame.shape[:2], dtype="uint8")
        img_triple_contours = np.zeros(frame.shape[:2], dtype="uint8")
        # img_outer_contours = np.zeros(frame.shape[:2], dtype="uint8")
        img_double_contours = np.zeros(frame.shape[:2], dtype="uint8")
        # img_all_contours = np.zeros(frame.shape, dtype="uint8")

        cv.drawContours(img_bulls_eye_contours, contours_bulls_eye, -1, (255, 255, 255, -1))
        cv.drawContours(img_bull_contours, contours_bull, -1, (255, 255, 255, -1))
        # cv.drawContours(img_inner_contours, contours_inner, -1, (255, 255, 255, -1))
        cv.drawContours(img_triple_contours, contours_triple, -1, (255, 255, 255, -1))
        # cv.drawContours(img_outer_contours, contours_outer, -1, (255, 255, 255, -1))
        cv.drawContours(img_double_contours, contours_double, -1, (255, 255, 255, -1))

        # cv.drawContours(img_all_contours, contours_bull, -1, (0, 255, 0), -1)
        # cv.drawContours(img_all_contours, contours_bulls_eye, -1, (0, 0, 255), -1)
        # cv.drawContours(img_all_contours, contours_inner, -1, (255, 0, 0), -1)
        # cv.drawContours(img_all_contours, contours_triple, -1, (0, 255, 255), -1)
        # cv.drawContours(img_all_contours, contours_outer, -1, (255, 0, 255), -1)
        # cv.drawContours(img_all_contours, contours_double, -1, (255, 255, 0), -1)

        cv.imshow("bulls eye", img_bulls_eye_contours)
        cv.imshow("bull", img_bull_contours)
        # cv.imshow("inner", img_inner_contours)
        cv.imshow("triple", img_triple_contours)
        # cv.imshow("outer", img_outer_contours)
        cv.imshow("double", img_double_contours)
        # cv.imshow("all contours", img_all_contours)

        cv.imshow("equalized", equalized_bgr)

        # gradient_values = np.gradient(b, 1)
        # plt.plot(gradient_values, 'b')
        # plt.show()
        # cv.drawContours(frame, convex_hull_triple, -1, (0, 0, 255), 2)

        cv.imshow("sliders", frame)

        frame_time_end = time.time_ns()
        print(f"Frame time: {round((frame_time_end - frame_time) / 1000000, 2)} ms")

        rval, frame = cam.read()

        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            cv.destroyAllWindows()
            break


def correct_triple_and_double_contour_categories(bulls_eye_cp, contours_triple, contours_double):
    # correct placement of triple and double segments, sometimes they were matched in the wrong category

    padc = []  # PointAngleDistanceContour
    for contour in contours_double + contours_triple:
        cp = get_center_of_contour(contour)
        angle = get_angle(cp, bulls_eye_cp)
        distance = get_distance(cp, bulls_eye_cp)
        padc.append(CoordinateAngleDistanceContour(cp, angle, distance, contour))
    # put the instances all in a list, sorted by angle
    padc.sort(key=lambda x: x.angle)
    # make a savgol filter with big smoothing parameter (like 10) to smooth the distance curve
    savgol_curve = savgol_filter([x.distance for x in padc], 11, 1)
    # plt.plot([cadc.angle for cadc in padc], [cadc.distance for cadc in padc])
    # plt.plot([cadc.angle for cadc in padc], savgol_curve)
    # plt.show()
    # sort the contours by distance above and below savgol curve
    contours_triple = []
    contours_double = []
    for i in list(range(0, len(padc))):
        if savgol_curve[i] > padc[i].distance:
            contours_triple.append(padc[i].contour)
        else:
            contours_double.append(padc[i].contour)
    return contours_triple, contours_double


def get_smoothed_ring_contours(ring_contours: List[np.ndarray], img_width, img_height, bulls_eye_cp):
    inner_ring_edge_values: List[EdgeValues] = []
    outer_ring_edge_values: List[EdgeValues] = []
    for cnt in ring_contours:
        cads = get_cads(cnt, bulls_eye_cp)
        cnt_inner_ring_coordinates, cnt_outer_ring_coordinates = get_edge_values(cads, 0.5, 0.5, 0.1, 0.1)
        inner_ring_edge_values.append(cnt_inner_ring_coordinates)
        outer_ring_edge_values.append(cnt_outer_ring_coordinates)
    inner_ring_interpolation, outer_ring_interpolation = get_interpolations(inner_ring_edge_values, outer_ring_edge_values, img_width, img_height, bulls_eye_cp)

    inner_ring_smoothed_contour = get_endpoint_range(bulls_eye_cp, inner_ring_interpolation.angles, inner_ring_interpolation.distances, safe_angle_and_distance=False)
    inner_ring_smoothed_contour = np.asarray([(c.x, c.y) for c in inner_ring_smoothed_contour])
    outer_ring_smoothed_contour = get_endpoint_range(bulls_eye_cp, outer_ring_interpolation.angles, outer_ring_interpolation.distances, safe_angle_and_distance=False)
    outer_ring_smoothed_contour = np.asarray([(c.x, c.y) for c in outer_ring_smoothed_contour])

    return inner_ring_smoothed_contour, outer_ring_smoothed_contour


def get_angle_between_ring_elements(bulls_eye_cp, ring_element_with_next_smaller_angle, ring_element_with_next_bigger_angle):
    smaller_angle_contour_cads = get_cads(ring_element_with_next_smaller_angle.contour, bulls_eye_cp)
    smaller_contour_cads_angles = np.asarray([cad.angle for cad in smaller_angle_contour_cads])
    if min(smaller_contour_cads_angles) < 1 and max(smaller_contour_cads_angles) > 5:
        smaller_max_angle = max(smaller_contour_cads_angles[np.argwhere(smaller_contour_cads_angles < 5)])[0]
    else:
        smaller_max_angle = max(smaller_contour_cads_angles)

    bigger_angle_contour_cads = get_cads(ring_element_with_next_bigger_angle.contour, bulls_eye_cp)
    bigger_contour_cads_angles = np.asarray([cad.angle for cad in bigger_angle_contour_cads])
    if min(bigger_contour_cads_angles) < 1 and max(bigger_contour_cads_angles) > 5:
        bigger_min_angle = min(bigger_contour_cads_angles[np.argwhere(bigger_contour_cads_angles > 5)])[0]
    else:
        bigger_min_angle = min(bigger_contour_cads_angles)
    contour_mid_angle = get_mid_angle(smaller_max_angle, bigger_min_angle)

    return contour_mid_angle


def get_adjacent_ring_elements(cadcs_triple_or_double: List[CoordinateAngleDistanceContour], segments_side_coordinate_group):
    triple_angles = np.asarray([cadc.angle for cadc in cadcs_triple_or_double], dtype=np.float32)
    indices_of_smaller_triple_angles = np.argwhere(triple_angles < segments_side_coordinate_group.mean_angle)
    if indices_of_smaller_triple_angles.size == 0:
        i_next_smaller_triple = len(triple_angles) - 1
    else:
        i_next_smaller_triple = indices_of_smaller_triple_angles[-1][0]
    triple_with_next_smaller_angle = cadcs_triple_or_double[i_next_smaller_triple]
    indices_of_bigger_triple_angles = np.argwhere(triple_angles > segments_side_coordinate_group.mean_angle)
    if indices_of_bigger_triple_angles.size == 0:
        i_next_bigger_triple = 0
    else:
        i_next_bigger_triple = indices_of_bigger_triple_angles[0][0]
    triple_with_next_bigger_angle = cadcs_triple_or_double[i_next_bigger_triple]
    return triple_with_next_smaller_angle, triple_with_next_bigger_angle


def get_cadcs_for_contours(board_cp, contours) -> List[CoordinateAngleDistanceContour]:
    cadcs = []
    for cnt in contours:
        cnt_c = get_center_of_contour(cnt)
        angle = get_angle(cnt_c, board_cp)
        distance = get_distance(board_cp, cnt_c)
        cadcs.append(CoordinateAngleDistanceContour(cnt_c, angle, distance, cnt))

    return cadcs


def smooth_transition_point_groups(bulls_eye_cp, radial_section_transitions) -> List[List[CoordinateAngleDistance]]:
    smoothed_transition_point_groups = []
    for coordinate_group in radial_section_transitions:
        coordinate_group.sort(key=lambda x: x.distance)

        angles = [cad.angle for cad in coordinate_group]
        distances = [cad.distance for cad in coordinate_group]

        angle_diffs = get_angle_diffs(angles)
        angle_diff_cutoff = (2 * np.pi) / 400
        abs_angle_diffs = np.asarray([abs(diff) for diff in angle_diffs])
        bad_angles_indices = np.argwhere(abs_angle_diffs > angle_diff_cutoff)
        if bad_angles_indices.size > 0:
            bad_angles_indices = bad_angles_indices + 1  # np.argwhere returns one i earlier then the actual bad i
            bad_angles_indices = [i[0] for i in bad_angles_indices]
            # remove bad angles that are too far from the angles around them
            [angles.pop(i) for i in bad_angles_indices if i < len(angles)]
            [distances.pop(i) for i in bad_angles_indices if i < len(distances)]

        new_coordinates: List[CoordinateAngleDistance] = get_interpolated_coordinates_with_smoothed_angles(angles, distances, bulls_eye_cp, 100, 11)
        smoothed_transition_point_groups.append(new_coordinates)

    return smoothed_transition_point_groups


def get_angles_shifted_to_safe_range(angles):
    if min(angles) < 1 and max(angles) > 5:
        big_angles = [angle for angle in angles if angle > 5]
        small_angles = [angle for angle in angles if angle < 1]
        angle_shift = (2 * np.pi) - min(big_angles)
        big_angles = [(angle + angle_shift) % (2 * np.pi) for angle in big_angles]
        small_angles = [angle + angle_shift for angle in small_angles]
        angles_shifted = big_angles + small_angles
    else:
        angles_shifted = angles
        angle_shift = 0

    return angles_shifted, angle_shift


def shift_angles_back_to_original_range(angles, angle_shift):
    angles_shifted_back = [(angle - angle_shift) % (2 * np.pi) for angle in angles]
    angles_negative = [angle for angle in angles_shifted_back if angle < 0]
    angles_negative = [(2 * np.pi) + angle for angle in angles_negative]  # its addition because values are negative
    angles_positive = [angle for angle in angles_shifted_back if angle >= 0]
    angles_shifted_correctly = angles_negative + angles_positive
    return angles_shifted_correctly


def get_angle_diffs(angles):
    angles_shifted, _ = get_angles_shifted_to_safe_range(angles)

    return np.diff(angles_shifted).astype(np.float32)


def get_interpolated_coordinates_with_smoothed_angles(angles, distances, base_coordinate: Coordinate2d, coordinate_quantity: int, savgol_window_length: int) -> List[CoordinateAngleDistance]:
    angles_shifted, angle_shift = get_angles_shifted_to_safe_range(angles)
    distance_linspace = np.linspace(distances[0], distances[-1], coordinate_quantity)
    angles_interpolated = np.interp(distance_linspace, distances, angles_shifted)
    smoothed_angles = savgol_filter(angles_interpolated, savgol_window_length, 1, mode='mirror')
    smoothed_angles_shifted_back = shift_angles_back_to_original_range(smoothed_angles, angle_shift)
    new_coordinates: List[CoordinateAngleDistance] = get_endpoint_range(base_coordinate, smoothed_angles_shifted_back, distance_linspace)

    return new_coordinates


def group_cads_by_angle(cads, angle_diff_cutoff) -> List[CoordinateGroup]:
    # groups unsorted points by angle to bull's eye center point
    cads.sort(key=lambda x: x.angle)
    angle_diffs = np.diff([itp.angle for itp in cads])
    # identifies angle groups with low angle differences
    masked_groups = np.ma.masked_where(angle_diffs > angle_diff_cutoff, angle_diffs)
    masked_group_slices = np.ma.flatnotmasked_contiguous(masked_groups)  # get index slices of masked groups
    point_groups = []
    for point_slice in masked_group_slices:
        group = cads[point_slice.start:point_slice.stop]
        point_groups.append(group)

    if len(point_groups[0]) < 8 and len(point_groups[-1]) < 8 and len(point_groups) == 21:
        # case where a transition group is split in half because of the radian limit
        last_group = point_groups[-1] + point_groups[0]
        point_groups = point_groups[1:] + [last_group]

    point_groups = [CoordinateGroup(pg) for pg in point_groups]

    return point_groups


def get_coordinate_group_pairs_by_angle(inner_coordinate_groups: List[CoordinateGroup], outer_coordinate_groups: List[CoordinateGroup]) -> List[Tuple[CoordinateGroup]]:
    # get pairs by angle, one pair is an inner+outer white to black transition coordinate group
    inner_coordinate_groups.sort(key=lambda x: x.mean_angle)
    outer_coordinate_groups.sort(key=lambda x: x.mean_angle)
    first_inner_angle = inner_coordinate_groups[0].mean_angle
    first_outer_angle = outer_coordinate_groups[0].mean_angle
    first_angle_diff = abs(first_inner_angle - first_outer_angle)
    if not first_angle_diff < 0.1:
        if first_inner_angle < first_outer_angle:
            outer_coordinate_groups = [outer_coordinate_groups[-1]] + outer_coordinate_groups[:-1]
        else:
            outer_coordinate_groups = outer_coordinate_groups[1:] + [outer_coordinate_groups[0]]

    coordinate_groups = list(zip(inner_coordinate_groups, outer_coordinate_groups))

    return coordinate_groups


def get_ellipse_range(x1, y1, a1, b1, angle1, x2, y2, a2, b2, angle2, ellipse_quantity, img) -> List[np.ndarray]:
    additional_small_ellipses = 2
    additional_big_ellipses = 8
    all_ellipses_quantity = ellipse_quantity + additional_small_ellipses + additional_big_ellipses
    xy_linspace = np.linspace((x1, y1), (x2, y2), all_ellipses_quantity, dtype=(np.uint16, np.uint16))
    ab_linspace = np.linspace((a1 / 2, b1 / 2), (a2 / 2, b2 / 2), all_ellipses_quantity, dtype=(np.uint16, np.uint16))
    angle_linspace = np.linspace(angle1, angle2, all_ellipses_quantity)
    xy_linspace = xy_linspace[additional_small_ellipses:-additional_big_ellipses]
    ab_linspace = ab_linspace[additional_small_ellipses:-additional_big_ellipses]
    angle_linspace = angle_linspace[additional_small_ellipses:-additional_big_ellipses]
    discrete_ellipses = []
    coordinate_quantity = 2000
    for (x, y), (a, b), angle in list(zip(xy_linspace, ab_linspace, angle_linspace)):
        cv.ellipse(img, (int(x), int(y)), (int(a), int(b)), int(angle), 0, 360, (255, 255, 255), 1)
        ellipse_points = cv.ellipse2Poly((int(x), int(y)), (int(a), int(b)), angle=int(angle), arcStart=0, arcEnd=360, delta=1)
        x_lin = np.linspace(0, len(ellipse_points) - 1, num=len(ellipse_points), dtype=np.uint16)
        y_lin = ellipse_points
        xnew = np.linspace(0, len(ellipse_points) - 1, num=coordinate_quantity, dtype=np.uint16)
        ylin_x = np.interp(xnew, x_lin, y_lin[:, 0]).astype(np.uint16)
        ylin_y = np.interp(xnew, x_lin, y_lin[:, 1]).astype(np.uint16)
        ylin = np.column_stack([ylin_x, ylin_y])
        discrete_ellipses.append(ylin)

    # cv.ellipse(img, (int(x1), int(y1)), (int(a1 / 2), int(b1 / 2)), int(angle1), 0, 360, (0, 0, 255), 1)
    # cv.ellipse(img, (int(x2), int(y2)), (int(a2 / 2), int(b2 / 2)), int(angle2), 0, 360, (0, 255, 0), 1)
    # cv.imshow("ellipse img", img)

    return discrete_ellipses


def get_black_white_transition_points(equalized_bgr, convex_hull_small, convex_hull_big, bulls_eye_cp) -> List[CoordinateAngleDistance]:
    (x1, y1), (a1, b1), angle1 = cv.fitEllipse(convex_hull_small)
    (x2, y2), (a2, b2), angle2 = cv.fitEllipse(convex_hull_big)
    ellipse_quantity = 20
    discrete_ellipses = get_ellipse_range(x1, y1, a1, b1, angle1, x2, y2, a2, b2, angle2, ellipse_quantity, copy.deepcopy(equalized_bgr))

    all_black_white_transition_points: List[CoordinateAngleDistance] = []
    for ellipse_points in discrete_ellipses:
        ellipse_points_bgr_values = equalized_bgr[ellipse_points[:, 1], ellipse_points[:, 0]]
        mean_ellipse_values = np.mean(ellipse_points_bgr_values, axis=1)

        mean_ellipse_values_tripled = np.concatenate([mean_ellipse_values, mean_ellipse_values, mean_ellipse_values])
        tripled_start_index = len(mean_ellipse_values)
        tripled_end_index = len(mean_ellipse_values) * 2

        window_length = int(len(ellipse_points) * 0.231)
        if window_length % 2 == 0:
            window_length += 1
        smoothed_transition_curve = savgol_filter(mean_ellipse_values_tripled, window_length, 3)

        masked_above_avg = np.ma.masked_where(mean_ellipse_values_tripled < smoothed_transition_curve - 1, mean_ellipse_values_tripled)
        mask_edge_slices_above = np.ma.flatnotmasked_contiguous(masked_above_avg)
        edges_above = [es for es in mask_edge_slices_above if es.stop - es.start > 20]
        edge_indices_above = np.ravel([(es.start, es.stop - 1) for es in edges_above])
        edge_indices_above = [i - tripled_start_index for i in edge_indices_above if tripled_start_index < i < tripled_end_index]
        mask_transitions_above = np.full(len(mean_ellipse_values), fill_value=np.NaN)
        mask_transitions_above[edge_indices_above] = mean_ellipse_values[edge_indices_above]

        masked_below_avg = np.ma.masked_where(mean_ellipse_values_tripled > smoothed_transition_curve + 1, mean_ellipse_values_tripled)
        mask_edge_slices_below = np.ma.flatnotmasked_contiguous(masked_below_avg)
        edges_below = [es for es in mask_edge_slices_below if es.stop - es.start > 20]
        edge_indices_below = np.ravel([(es.start, es.stop - 1) for es in edges_below])
        edge_indices_below = [i - tripled_start_index for i in edge_indices_below if tripled_start_index < i < tripled_end_index]
        mask_transitions_below = np.full(len(mean_ellipse_values), fill_value=np.NaN)
        mask_transitions_below[edge_indices_below] = mean_ellipse_values[edge_indices_below]

        if len(edge_indices_above) == 20 and len(edge_indices_below) == 20:
            mean_transition_indices = np.mean([edge_indices_above, edge_indices_below], axis=0, dtype=np.uint32)
            ellipse_black_white_transition_points = ellipse_points[mean_transition_indices]
            ebwt_cads = get_cads(ellipse_black_white_transition_points, bulls_eye_cp)
            all_black_white_transition_points.extend(ebwt_cads)

        # debug plots
        # plt.plot(smoothed_transition_curve[tripled_start_index:tripled_end_index], label="smoothed transition curve")
        # plt.plot(np.ma.filled(masked_above_avg[tripled_start_index:tripled_end_index], np.NaN), label="edge slices below")
        # plt.plot(np.ma.filled(masked_below_avg[tripled_start_index:tripled_end_index], np.NaN), label="transitions above")
        # plt.plot(mask_transitions_above, 'x', label=f"transitions above ({len(edge_indices_above)})")
        # plt.plot(mask_transitions_below, 'x', label=f"transitions below ({len(edge_indices_below)})")
        # plt.legend()
        # plt.show()
        # x = 0

    return all_black_white_transition_points


threshold_sliders()
