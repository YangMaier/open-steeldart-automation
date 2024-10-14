from typing import List, Tuple

import cv2 as cv
import numpy as np
from scipy.signal import savgol_filter

from data_structures.board_segment import BoardSegment
from data_structures.coordinates import Coordinate2d, Coordinate2dNormalized
from data_structures.coordinates_extended import CoordinateAngleDistanceContour, CoordinateAngleDistance
from data_structures.interpolation import EdgeValues, AngleDistanceInterpolation
from operations_on.angles import pol2cart, get_mid_angle, get_min_angle, get_max_angle, is_angle_in_range
from operations_on.coordinates import get_distance, get_interpolation, get_points_as_nparray
from fiddling.misc import sort_points_clockwise
from operations_on.coordinates_and_angles import get_cads, get_edge_values, get_endpoint_range, get_angle
from operations_on.ellipses import get_fit_ellipse
from operations_on.lists import flatten, flatten_np_array_lists


def get_contour_area(contour) -> float:
    return cv.contourArea(contour)


def get_contour_area_normalized(contour, img_width, img_height) -> float:
    return round(get_contour_area(contour) / (img_width * img_height), 6)


def filter_center_inside_contour(contour):
    try:
        cp = get_center_of_contour(contour)
        dist = cv.pointPolygonTest(contour, (cp.x, cp.y), measureDist=False)
    except:
        print("Error in filter_center_inside_contour, returned False")
        return False
    return dist >= 0  # contour center is inside the contour


def get_center_of_contour(contour) -> Coordinate2d:
    contour = np.asarray(contour, dtype=np.int32)
    m = cv.moments(contour)
    try:
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
    except ZeroDivisionError as zde:
        cx = int(contour[0][0][0])
        cy = int(contour[0][0][1])
    return Coordinate2d(cx, cy)


def get_center_of_contour_normalized(contour, img_width, img_height) -> Coordinate2dNormalized:
    cp = get_center_of_contour(contour)
    return Coordinate2dNormalized(cp.x / img_width, cp.y / img_height)


def get_normalized_contour(contour, img_width, img_height) -> np.ndarray:
    return np.array([(round(point[0] / img_width, 6), round(point[1] / img_height, 6)) for point in contour])


def get_contour_from_normalized(contour_normalized, img_width, img_height) -> np.ndarray:
    return np.array([(int(point.x * img_width), int(point.y * img_height)) for point in contour_normalized])


def get_area_from_normalized_area(area_normalized: float, img_width: int, img_height: int) -> float:
    return area_normalized * (img_width * img_height)


def get_center_of_contour_precise(contour) -> tuple:
    m = cv.moments(contour)
    try:
        cx = m['m10'] / m['m00']
        cy = m['m01'] / m['m00']
    except ZeroDivisionError as zde:
        cx = contour[0][0][0]
        cy = contour[0][0][1]
    return round(cx, 2), round(cy, 2)


def filter_min_size_contour(contour, cnt_min_area):
    return cv.contourArea(contour) > cnt_min_area


def filter_max_size_contour(contour, cnt_max_area):
    return cv.contourArea(contour) < cnt_max_area


def get_contour_solidity(contour):
    area = cv.contourArea(contour)
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    if hull_area == 0:
        solidity = 0
    else:
        solidity = float(area) / hull_area
    return round(solidity, 6)


def filter_contour_solidity(contour, solidity_threshold=0.6):
    # Solidity is the ratio of contour area to its convex hull area
    solidity = get_contour_solidity(contour)
    return solidity > solidity_threshold


def filter_contour_min_rotated_extent(contour, extend_threshold):
    extent = get_contour_rotated_extent(contour)
    return extent > extend_threshold


def filter_contour_max_rotated_extent(contour, extend_threshold):
    extent = get_contour_rotated_extent(contour)
    return extent < extend_threshold


def get_contour_rotated_extent(contour):
    area = cv.contourArea(contour)
    # x, y, w, h = cv.boundingRect(contour)
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box_pixels = np.array([[int(p[0]), int(p[1])] for p in box])
    rotated_rectangle_area = cv.contourArea(box_pixels)

    # rect_area = w * h
    if rotated_rectangle_area == 0:
        extent = 0
    else:
        extent = float(area) / rotated_rectangle_area  # max is 1

    return round(extent, 6)


def get_elongation_ratio(cnt):
    (_, _), (width, height), angle = cv.minAreaRect(cnt)

    # test_img = get_test_img_for_contour(cnt)

    if min(width, height) <= 0:

        return 0  # Avoid division by zero. Is not a segment anyway with height or width 0

    elongation = round(max(width, height) / min(width, height), 2)

    return elongation


def filter_min_elongation(cnt, elongation_ratio):
    elongation = get_elongation_ratio(cnt)
    return elongation > elongation_ratio


def filter_max_elongation(cnt, elongation_ratio):
    elongation = get_elongation_ratio(cnt)
    return elongation < elongation_ratio


def get_test_img_for_contour(cnt):
    test_img = np.zeros(
        [max(cnt[:, 0][:, 1]) + 2 - min(cnt[:, 0][:, 1]), max(cnt[:, 0][:, 0]) + 2 - min(cnt[:, 0][:, 0])],
        dtype=np.uint8)
    cnt_min_x = min(cnt[:, 0][:, 0])
    cnt_min_y = min(cnt[:, 0][:, 1])
    draw_cnt = cnt.copy()
    draw_cnt[:, 0][:, 0] = cnt[:, 0][:, 0] - cnt_min_x
    draw_cnt[:, 0][:, 1] = cnt[:, 0][:, 1] - cnt_min_y
    cv.drawContours(test_img, [draw_cnt], -1, (255, 255, 255), 1)

    return test_img


def get_test_img_for_cads(cads):
    contour = np.asarray([[[cad.coordinate.x, cad.coordinate.y]] for cad in cads], dtype=np.int32)
    return get_test_img_for_contour(contour)


def is_elongated(cnt, length_ratio):
    rect = cv.minAreaRect(cnt)
    # rect[0] is center x,y
    # rect[1] is width and height
    if min(rect[1]) <= 0:
        return False  # Avoid division by zero. Is not a double field anyway with height or width 0
    if (max(rect[1]) / min(rect[1])) > (length_ratio):
        return True
    else:
        return False


def filter_length_ratio(cnt, length_ratio):
    rect = cv.minAreaRect(cnt)
    # rect[0] is center x,y
    # rect[1] is width and height
    if min(rect[1]) <= 0:
        return False  # Avoid division by zero. Is not a double field anyway with height or width 0
    if (max(rect[1]) / min(rect[1])) > (length_ratio):
        return True
    else:
        return False


def get_extended_rect(cnt, width_extension_rate=1.2, height_extension_rate=1.2):
    # calculates a rotated bounding rectangle that is longer than the original
    (x, y), (w, h), angle = cv.minAreaRect(cnt)
    if w != 0 and h != 0:
        if w > h:
            w *= width_extension_rate
            h *= height_extension_rate
        if h > w:
            h *= width_extension_rate
            w *= height_extension_rate
    box = cv.boxPoints(((x, y), (w, h), angle))
    box = np.int0(box)
    return box


def is_double_field(contour, double_segment_min_size, double_segment_max_size):
    center = get_center_of_contour(contour)
    c1 = filter_min_size_contour(contour, double_segment_min_size)
    c2 = filter_max_size_contour(contour, double_segment_max_size)
    c3 = filter_contour_min_rotated_extent(contour, 0.6)
    c4 = is_elongated(contour, 2.9)
    c5 = filter_center_inside_contour(contour, center)

    return c1 and c2 and c3 and c4 and c5


def is_probably_a_board_field(contour, board_segment_min_size, board_segment_max_size):
    return (
            filter_min_size_contour(contour, board_segment_min_size) and
            filter_max_size_contour(contour, board_segment_max_size) and
            filter_center_inside_contour(contour, get_center_of_contour(contour)) and
            filter_contour_solidity(contour)
    )


def scale_contour(cnt, scale):
    # https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def rotate_contour(cnt, angle):
    # https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
    M = cv.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


def match_contour(live_contour, sequence_contour) -> float or None:
    # avoid this function if possible. It's return value is inconsistent.e don't want to use it.
    if live_contour is None or sequence_contour is None:
        return None
    else:
        match: float = cv.matchShapes(live_contour, sequence_contour, 3, 0.0)
        if match > 1000:
            # not viable, matchShapes seems to have some consistency issues, and we don't need values above 1000
            return 1000
        return round(match, 6)


def get_contour_distance_to(contour, point: Coordinate2d):
    # returns negative number if point is outside the contour
    # returns 0 if point is on the contour
    # returns positive number if point is inside the contour
    return cv.pointPolygonTest(contour, (point.x, point.y), measureDist=True)


def get_approximated_line_on_contour(img, contour):
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)

    return cols, lefty, righty


def get_impact_point(img_empty_board: np.ndarray, img_one_dart: np.ndarray):

    # impact point = average those points
    img_diff = cv.absdiff(img_empty_board, img_one_dart)

    all_contours_points_clockwise_cv, contours = get_dart_contour(img_empty_board, img_one_dart)

    drawn_img = np.zeros(img_diff.shape[:2], dtype="uint8")
    cv.drawContours(drawn_img, contours, -1, 255, 1)
    cv.drawContours(drawn_img, all_contours_points_clockwise_cv, -1, 255, 1)

    x, y, w, h = cv.boundingRect(all_contours_points_clockwise_cv)
    cv.rectangle(drawn_img, (x, y), (x + w, y + h), 255, 1)

    convex_hull = cv.convexHull(all_contours_points_clockwise_cv)

    max_y_index = np.argmax(convex_hull[:, 0][:, 1], axis=0)
    y_max_xy = convex_hull[max_y_index][0]
    p_y_max = Coordinate2d(y_max_xy[0], y_max_xy[1])

    distances_to_p_y_max = [
        [
            get_distance(Coordinate2d(p[0][0], p[0][1]), p_y_max),
            Coordinate2d(p[0][0], p[0][1])
        ]
        for p in convex_hull
    ]

    nc_ch_point_contour_distances = [p for p in distances_to_p_y_max if p[0] < 30]
    mean_point = np.mean([[p[1].x, p[1].y] for p in nc_ch_point_contour_distances], axis=0)
    impact_point = Coordinate2d(int(mean_point[0]), int(mean_point[1]))

    cv.circle(img_diff, (impact_point.x, impact_point.y), 5, (255, 255, 255), 1)

    return impact_point


def get_dart_contour(last_frame, frame):
    img_diff = cv.absdiff(last_frame, frame)
    grey = cv.cvtColor(img_diff, cv.COLOR_BGR2GRAY)
    thresh_diff = cv.threshold(grey, 8, 255, cv.THRESH_BINARY)[1]
    opening = cv.morphologyEx(thresh_diff, cv.MORPH_OPEN, np.ones((3, 3)))
    contours, _ = cv.findContours(opening, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # filter for contour sizes that are big enough
    if img_diff.shape[0] == 1080 or img_diff.shape[0] == 1920:
        contour_area_filter = 1000
    elif img_diff.shape[0] == 640 or img_diff.shape[0] == 480:
        contour_area_filter = 300
    else:
        raise NotImplementedError
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > contour_area_filter]
    # relevant_contours = [cnt for cnt in contours[0] if cv.contourArea(cnt) > 30]
    # get convex hull of these contours for a complete dart shape
    all_contour_points = []
    for i in range(0, len(contours)):
        for p in contours[i]:
            all_contour_points.append(p[0])

    if not all_contour_points:

        return None, None

    center_pt = np.array(all_contour_points).mean(axis=0)
    clock_ang_dist = sort_points_clockwise.ClockwiseAngleAndDistance(center_pt)
    all_contours_clockwise = sorted(all_contour_points, key=clock_ang_dist)
    all_contours_points_clockwise_cv = (np.array(all_contours_clockwise).reshape((-1, 1, 2)).astype(np.int32))
    return all_contours_points_clockwise_cv, contours


def get_smoothed_ring_cads_somewhat_working(ring_contours: List[np.ndarray], img_width, img_height, board_center_coordinate: Coordinate2d) -> Tuple[List[CoordinateAngleDistance], List[CoordinateAngleDistance]]:
    inner_ring_edge_values: List[EdgeValues] = []
    outer_ring_edge_values: List[EdgeValues] = []
    for cnt in ring_contours:
        cads = get_cads(cnt, board_center_coordinate)
        cnt_inner_ring_coordinates, cnt_outer_ring_coordinates = get_edge_values(cads, 0.5, 0.5, 0.1, 0.1)
        inner_ring_edge_values.append(cnt_inner_ring_coordinates)
        outer_ring_edge_values.append(cnt_outer_ring_coordinates)
    inner_ring_interpolation, outer_ring_interpolation = get_interpolations(inner_ring_edge_values, outer_ring_edge_values, img_width, img_height, board_center_coordinate)

    inner_ring_smoothed_cads = get_endpoint_range(board_center_coordinate, inner_ring_interpolation.angles, inner_ring_interpolation.distances)

    outer_ring_smoothed_cads = get_endpoint_range(board_center_coordinate, outer_ring_interpolation.angles, outer_ring_interpolation.distances)

    return inner_ring_smoothed_cads, outer_ring_smoothed_cads


def get_smoothed_ring_cads(ring_contours: List[np.ndarray], img_width, img_height, board_center_coordinate: Coordinate2d, img) -> Tuple[List[CoordinateAngleDistance], List[CoordinateAngleDistance]]:
    inner_ring_edge_values: List[EdgeValues] = []
    outer_ring_edge_values: List[EdgeValues] = []
    all_contours_np_points = flatten_np_array_lists(ring_contours)
    convex_hull = cv.convexHull(all_contours_np_points)
    convex_hull_cads = get_cads(convex_hull, board_center_coordinate)
    all_cads = get_cads(all_contours_np_points, board_center_coordinate)
    all_cads.sort(key=lambda x: x.angle)
    all_cads_filtered = []
    for cnt in ring_contours:
        cads = get_cads(cnt, board_center_coordinate)
        min_angle = get_min_angle([cad.angle for cad in cads])
        max_angle = get_max_angle([cad.angle for cad in cads])
        filtered_cads = [
            cad for cad in cads if
            is_angle_in_range(
                cad.angle,
                (min_angle + 0.03) % (2 * np.pi),
                (max_angle - 0.03) % (2 * np.pi)
            )
        ]
        all_cads_filtered.extend(filtered_cads)

    all_cads_filtered.sort(key=lambda x: x.angle)
    window_length = int(len(all_cads_filtered) / 30)
    if window_length % 2 == 0:
        window_length += 1
    savgol_curve = savgol_filter([cad.distance for cad in all_cads_filtered], window_length, 1, mode="mirror")
    inner_ring_cads: List[CoordinateAngleDistance] = []
    outer_ring_cads: List[CoordinateAngleDistance] = []
    for savgol_curve_val, cad in zip(savgol_curve, all_cads_filtered):
        if savgol_curve_val > cad.distance:
            inner_ring_cads.append(cad)
        else:
            outer_ring_cads.append(cad)

    angles_inner_ring, smoothed_distances_inner_ring = get_interpolation(inner_ring_cads)
    smoothed_cads_inner_ring = get_endpoint_range(board_center_coordinate, angles_inner_ring, smoothed_distances_inner_ring)

    angles_outer_ring, smoothed_distances_outer_ring = get_interpolation(outer_ring_cads)
    smoothed_cads_outer_ring = get_endpoint_range(board_center_coordinate, angles_outer_ring, smoothed_distances_outer_ring)

    return smoothed_cads_inner_ring, smoothed_cads_outer_ring
    # cnt_inner_ring_coordinates, cnt_outer_ring_coordinates = get_edge_values(cads, 0.5, 0.5, 0.1, 0.1)
    # inner_ring_edge_values.append(cnt_inner_ring_coordinates)
    # outer_ring_edge_values.append(cnt_outer_ring_coordinates)
    # inner_ring_interpolation, outer_ring_interpolation = get_interpolations(inner_ring_edge_values, outer_ring_edge_values, img_width, img_height, board_center_coordinate)
    #
    # inner_ring_smoothed_cads = get_endpoint_range(board_center_coordinate, inner_ring_interpolation.angles, inner_ring_interpolation.distances)
    #
    # outer_ring_smoothed_cads = get_endpoint_range(board_center_coordinate, outer_ring_interpolation.angles, outer_ring_interpolation.distances)

    # return inner_ring_smoothed_cads, outer_ring_smoothed_cads


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


def get_interpolations(
        inner_ring_edge_values: List[EdgeValues],
        outer_ring_edge_values: List[EdgeValues],
        img_width: int,
        img_height: int,
        board_cp: Coordinate2d
):
    inner_segments = sorted(inner_ring_edge_values, key=lambda s: s.mid_angle)
    all_inner_ring_cads = flatten([s.edge_points for s in inner_segments])
    inner_ring_angles, inner_ring_smoothed_distances = get_interpolation(all_inner_ring_cads, minimize=True)
    inner_ring_smoothed_points = get_endpoint_range(board_cp, inner_ring_angles, inner_ring_smoothed_distances, safe_angle_and_distance=False)
    inner_ring_smoothed_points = get_points_as_nparray(inner_ring_smoothed_points)

    inner_ring_ellipse_points = get_fit_ellipse(inner_ring_smoothed_points, img_width, img_height)
    inner_ring_ellipse_points = get_cads(inner_ring_ellipse_points, board_cp)
    inner_ring_smoothed_ellipse_angles, inner_ring_smoothed_ellipse_distances = get_interpolation(inner_ring_ellipse_points, minimize=True)

    inner_ring_mean_distances = np.mean(
        [
            inner_ring_smoothed_distances,
            inner_ring_smoothed_ellipse_distances
        ], axis=0
    )
    inner_ring_angle_distance_interpolation = AngleDistanceInterpolation(inner_ring_angles, inner_ring_mean_distances)

    outer_segments = sorted(outer_ring_edge_values, key=lambda s: s.mid_angle)
    all_outer_ring_coordinates = flatten([s.edge_points for s in outer_segments])
    outer_ring_angles, outer_ring_smoothed_distances = get_interpolation(all_outer_ring_coordinates, maximize=True)
    outer_ring_smoothed_coordinates = get_endpoint_range(board_cp, outer_ring_angles, outer_ring_smoothed_distances, safe_angle_and_distance=False)
    outer_ring_smoothed_coordinates = get_points_as_nparray(outer_ring_smoothed_coordinates)

    outer_ring_ellipse_coordinates = get_fit_ellipse(outer_ring_smoothed_coordinates, img_width, img_height)
    outer_ring_ellipse_coordinates = get_cads(outer_ring_ellipse_coordinates, board_cp)
    outer_ring_smoothed_ellipse_angles, outer_ring_smoothed_ellipse_distances = get_interpolation(outer_ring_ellipse_coordinates, maximize=True)

    outer_ring_mean_distances = np.mean(
        [
            outer_ring_smoothed_distances,
            outer_ring_smoothed_ellipse_distances
        ], axis=0
    )
    outer_ring_angle_distance_interpolation = AngleDistanceInterpolation(outer_ring_angles, outer_ring_mean_distances)

    return inner_ring_angle_distance_interpolation, outer_ring_angle_distance_interpolation


def get_cadcs_for_contours(board_center_coordinate, contours) -> List[CoordinateAngleDistanceContour]:
    cadcs = []
    for cnt in contours:
        cnt_c = get_center_of_contour(cnt)
        angle = get_angle(cnt_c, board_center_coordinate)
        distance = get_distance(board_center_coordinate, cnt_c)
        cadcs.append(CoordinateAngleDistanceContour(cnt_c, angle, distance, cnt))

    return cadcs


def get_as_board_segment(board_center_coordinate, contour, segment_type) -> BoardSegment:
    center_coordinate = get_center_of_contour(contour)
    center_angle = get_angle(center_coordinate, board_center_coordinate)
    center_distance = get_distance(board_center_coordinate, center_coordinate)
    center_cad = CoordinateAngleDistance(center_coordinate, center_angle, center_distance)
    cads = get_cads(contour, board_center_coordinate)
    min_angle = get_min_angle([cad.angle for cad in cads])
    max_angle = get_max_angle([cad.angle for cad in cads])
    low_angle_side_cad = [cad for cad in cads if cad.angle == min_angle][0]
    high_angle_side_cad = [cad for cad in cads if cad.angle == max_angle][0]
    board_segment = BoardSegment(center_cad, contour, cads, low_angle_side_cad, high_angle_side_cad, segment_type)

    return board_segment


def get_as_board_segments(board_center_coordinate, contours, segment_type) -> List[BoardSegment]:
    board_segments = [get_as_board_segment(board_center_coordinate, cnt, segment_type) for cnt in contours]

    return board_segments


def translate_contour(contour, coordinate: Coordinate2d):
    contour_translated = contour + (coordinate.x, coordinate.y)

    return contour_translated
