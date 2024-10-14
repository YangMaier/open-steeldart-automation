import math

from typing import List, Tuple

import cv2
import numpy as np

from data_structures.board_segment import BoardSegment
from data_structures.interpolation import AngleDistanceInterpolation, EdgeValues
from data_structures.coordinates import Coordinate2d
from data_structures.coordinates_extended import CoordinateAngleDistance
from data_structures.score_segments import SegmentType
from operations_on.angles import get_angle_diff, get_angle_range, is_angle_in_range, get_min_angle, get_max_angle, \
    get_mid_angle
from operations_on.coordinates import get_distance, get_mid_coordinate


def get_angle(point1: Coordinate2d, point2: Coordinate2d) -> np.float32:
    angle = math.atan2(point1.y - point2.y, point1.x - point2.x)
    angle_fixed = angle % (2 * math.pi)
    # if angle < 0:
    #     angle = 2 * math.pi + angle
    return np.float32(angle_fixed)


def get_endpoint_via(base_coordinate: Coordinate2d, angle, distance, safe_angle_and_distance: bool = False) -> Coordinate2d or CoordinateAngleDistance:
    (x2, y2) = (base_coordinate.x + distance * math.cos(angle), base_coordinate.y + distance * math.sin(angle))
    coordinate = Coordinate2d(np.int32(x2), np.int32(y2))
    if safe_angle_and_distance:
        return CoordinateAngleDistance(coordinate, angle, distance)
    else:
        return coordinate


def get_endpoint_via_with_floats(base: Tuple[float, float], angle: float, distance: float):
    (x2, y2) = (base[0] + distance * math.cos(angle), base[1] + distance * math.sin(angle))
    return x2, y2


def get_endpoint_via_interp(base: Coordinate2d, angle: float, a_d_interp_1: AngleDistanceInterpolation, a_d_interp_2: AngleDistanceInterpolation = None) -> Coordinate2d:
    if a_d_interp_2 is None:
        return get_endpoint_via(
            base,
            angle,
            np.interp(
                angle,
                a_d_interp_1.angles,
                a_d_interp_1.distances
            ),
            safe_angle_and_distance=True
        )
    else:
        return get_endpoint_via(
            base,
            angle,
            np.mean([
                np.interp(
                    angle,
                    a_d_interp_1.angles,
                    a_d_interp_1.distances
                ),
                np.interp(
                    angle,
                    a_d_interp_2.angles,
                    a_d_interp_2.distances
                )
            ]),
            safe_angle_and_distance=True
        )


def get_endpoint_range(base: Coordinate2d, angles, distances, safe_angle_and_distance=True) -> List[CoordinateAngleDistance] or List[Coordinate2d]:
    return [get_endpoint_via(base, angle, distance, safe_angle_and_distance=safe_angle_and_distance) for angle, distance in zip(angles, distances)]


def get_endpoint_range_with_distance_interpolation(
        base: Coordinate2d,
        angle_range: List[float],
        distance_interpolation_1: AngleDistanceInterpolation,
        distance_interpolation_2: AngleDistanceInterpolation = None
) -> List[Coordinate2d]:
    if distance_interpolation_2 is None:

        return [
            get_endpoint_via(
                base,
                angle,
                np.interp(angle, distance_interpolation_1.angles, distance_interpolation_1.distances),
                safe_angle_and_distance=True
            )
            for angle in angle_range
        ]
    else:
        return [
            get_endpoint_via(
                base,
                angle,
                np.mean([
                    np.interp(
                        angle,
                        distance_interpolation_1.angles,
                        distance_interpolation_1.distances
                    ),
                    np.interp(
                        angle,
                        distance_interpolation_2.angles,
                        distance_interpolation_2.distances
                    )
                ]),
                safe_angle_and_distance=True
            )
            for angle in angle_range
        ]


def get_pads(bulls_eye_cp, transition_points):
    transition_pads = []  # PointAngleDistances (pad)
    for unsorted_transition_point_list in transition_points:
        for p in unsorted_transition_point_list:
            p = Coordinate2d(p[0], p[1])
            transition_pads.append(
                CoordinateAngleDistance(
                    p,
                    get_angle(bulls_eye_cp, p),
                    get_distance(bulls_eye_cp, p)
                )
            )
    return transition_pads


def get_cad(coordinate: Coordinate2d, board_center_coordinate: Coordinate2d) -> CoordinateAngleDistance:
    return CoordinateAngleDistance(
        coordinate,
        get_angle(coordinate, board_center_coordinate),
        get_distance(coordinate, board_center_coordinate)
    )


def get_cads(np_array: np.ndarray, board_center_coordinate: Coordinate2d) -> List[CoordinateAngleDistance]:
    """Calculates the angle and distance to board center for each coordinate in the given contour"""
    try:
        coordinates = [Coordinate2d(xy[0][0], xy[0][1]) for xy in np_array]
    except IndexError:  # sometimes the dimension of the input array is something else than expected
        coordinates = [Coordinate2d(xy[0], xy[1]) for xy in np_array]
    cads = [get_cad(c, board_center_coordinate) for c in coordinates]

    return cads


def get_edge_values(cads: List[CoordinateAngleDistance], e_1_4_dist_cutoff: float, e_2_3_dist_cutoff: float, e_1_4_angle_cutoff: float, e_2_3_angle_cutoff: float) -> Tuple[EdgeValues, EdgeValues]:

    min_angle = min(cads, key=lambda p: p.angle).angle
    max_angle = max(cads, key=lambda p: p.angle).angle
    if min_angle < 1 and max_angle > 5:  # radian max "overflow"
        min_angle = min(
            [p for p in cads if p.angle > 4],
            key=lambda p: p.angle
        ).angle
        max_angle = max(
            [p for p in cads if p.angle < 2],
            key=lambda p: p.angle
        ).angle

    angle_diff = get_angle_diff(max_angle, min_angle)
    angle_linspace = get_angle_range(min_angle, max_angle, num=6)

    e_1_4_points = []  # edge with less distance to board center
    e_2_3_points = []  # edge with more distance to board center

    angle_part_1 = [
        p for p in cads if
        is_angle_in_range(p.angle, angle_linspace[0], angle_linspace[1])
    ]
    angle_part_2 = [
        p for p in cads if
        is_angle_in_range(p.angle, angle_linspace[1], angle_linspace[2])
    ]
    angle_part_3 = [
        p for p in cads if
        is_angle_in_range(p.angle, angle_linspace[2], angle_linspace[3])
    ]
    angle_part_4 = [
        p for p in cads if
        is_angle_in_range(p.angle, angle_linspace[3], angle_linspace[4])
    ]
    angle_part_5 = [
        p for p in cads if
        is_angle_in_range(p.angle, angle_linspace[4], angle_linspace[5])
    ]

    for angle_part in [angle_part_1, angle_part_2, angle_part_3, angle_part_4, angle_part_5]:
        min_distance = min(
            angle_part,
            key=lambda p: p.distance,
            default=CoordinateAngleDistance(Coordinate2d(0, 0), 0, 0)
        ).distance

        max_distance = max(
            angle_part,
            key=lambda p: p.distance,
            default=CoordinateAngleDistance(Coordinate2d(0, 0), 0, 0)
        ).distance

        distance_diff = max_distance - min_distance
        e_1_4_points += [
            p for p in angle_part if
            p.distance < min_distance + distance_diff * e_1_4_dist_cutoff
        ]
        e_2_3_points += [
            p for p in angle_part if
            p.distance > max_distance - distance_diff * e_2_3_dist_cutoff
        ]

    c_1_angle = get_min_angle([p.angle for p in e_1_4_points])
    c_4_angle = get_max_angle([p.angle for p in e_1_4_points])

    e_1_4_points = [
        p for p in e_1_4_points if
        is_angle_in_range(
            p.angle,
            min_angle + angle_diff * e_1_4_angle_cutoff,
            max_angle - angle_diff * e_1_4_angle_cutoff
        )
    ]

    c_2_angle = get_min_angle([p.angle for p in e_2_3_points])
    c_3_angle = get_max_angle([p.angle for p in e_2_3_points])

    e_2_3_points = [
        p for p in e_2_3_points if
        is_angle_in_range(
            p.angle,
            min_angle + angle_diff * e_2_3_angle_cutoff,
            max_angle - angle_diff * e_2_3_angle_cutoff
        )
    ]

    e_1_4 = EdgeValues(e_1_4_points, c_1_angle, c_4_angle)
    e_2_3 = EdgeValues(e_2_3_points, c_2_angle, c_3_angle)

    return e_1_4, e_2_3


def get_min_angle_coordinate(cads) -> Coordinate2d:
    min_angle = get_min_angle([cad.angle for cad in cads])
    min_angle_index = np.argwhere(np.array([cad.angle for cad in cads]) == min_angle)[0][0]
    coordinate = cads[min_angle_index].coordinate

    return coordinate


def get_max_angle_coordinate(cads) -> Coordinate2d:
    max_angle = get_max_angle([cad.angle for cad in cads])
    max_angle_index = np.argwhere(np.array([cad.angle for cad in cads]) == max_angle)[0][0]
    coordinate = cads[max_angle_index].coordinate
    return coordinate


def split_in_half_by_angle(cads: List[CoordinateAngleDistance], min_equal_min=False) -> Tuple[List[CoordinateAngleDistance], List[CoordinateAngleDistance]]:
    min_angle = get_min_angle([cad.angle for cad in cads])
    max_angle = get_max_angle([cad.angle for cad in cads])
    mean_angle = get_mid_angle(min_angle, max_angle)
    low_angle_half = [cad for cad in cads if is_angle_in_range(cad.angle, min_angle, mean_angle)]
    high_angle_half = [cad for cad in cads if is_angle_in_range(cad.angle, mean_angle, max_angle)]

    return low_angle_half, high_angle_half


def get_mid_cad(low_angle_segment: BoardSegment, high_angle_segment: BoardSegment, board_center_coordinate) -> CoordinateAngleDistance:
    angle1 = get_max_angle([cad.angle for cad in low_angle_segment.contour_cads])
    angle2 = get_min_angle([cad.angle for cad in high_angle_segment.contour_cads])
    mid_angle = get_mid_angle(angle1, angle2)
    mid_distance = (low_angle_segment.center_cad.distance + high_angle_segment.center_cad.distance) / 2

    endpoint = get_endpoint_via(board_center_coordinate, mid_angle, mid_distance, safe_angle_and_distance=True)

    return endpoint


def split_in_half_by_distance(cads: List[CoordinateAngleDistance]) -> Tuple[List[CoordinateAngleDistance], List[CoordinateAngleDistance]]:
    min_distance = min([cad.distance for cad in cads])
    max_distance = max([cad.distance for cad in cads])
    mean_distance = (min_distance + max_distance) / 2
    low_distance_half = [cad for cad in cads if cad.distance <= mean_distance]
    high_distance_half = [cad for cad in cads if cad.distance >= mean_distance]

    return low_distance_half, high_distance_half




# def get_inner_or_outer_segment_corners(cads: List[CoordinateAngleDistance]) -> RadialSectionRingIntersection:
#     low_distance_half, high_distance_half = split_in_half_by_distance(cads)
#     ld_low_angle_quarter, ld_high_angle_quarter = split_in_half_by_angle(low_distance_half)
#     hd_low_angle_quarter, hd_high_angle_quarter = split_in_half_by_angle(high_distance_half)
#
#     ld_low_angle_quarter, hd_low_angle_quarter, hd_high_angle_quarter, ld_high_angle_quarter = get_segment_corners(hd_high_angle_quarter, hd_low_angle_quarter, ld_high_angle_quarter, ld_low_angle_quarter)
#     corners = RadialSectionRingIntersection(ld_low_angle_quarter, hd_low_angle_quarter, hd_high_angle_quarter, ld_high_angle_quarter)
#
#     return corners
#
#
# def get_triple_or_double_segment_corners(cads: List[CoordinateAngleDistance]) -> RadialSectionRingIntersection:
#     from operations_on.contours import get_test_img_for_cads
#     low_angle_half, high_angle_half = split_in_half_by_angle(cads)
#     cad_img = get_test_img_for_cads(cads)
#     la_img = get_test_img_for_cads(low_angle_half)
#     low_angle_half, _ = split_in_half_by_angle(low_angle_half)
#     _, high_angle_half = split_in_half_by_angle(high_angle_half)
#     la_low_distance_half, la_high_distance_half = split_in_half_by_distance(low_angle_half)
#     ha_low_distance_half, ha_high_distance_half = split_in_half_by_distance(high_angle_half)
#
#     ld_low_angle_quarter, hd_low_angle_quarter, hd_high_angle_quarter, ld_high_angle_quarter = get_segment_corners(
#         la_low_distance_half, la_high_distance_half, ha_high_distance_half, ha_low_distance_half)
#     corners = RadialSectionRingIntersection(ld_low_angle_quarter, hd_low_angle_quarter, hd_high_angle_quarter, hd_low_angle_quarter)
#
#     return corners
#
#
# def get_segment_corners(ld_low_angle_quarter, hd_low_angle_quarter, hd_high_angle_quarter, ld_high_angle_quarter) -> Tuple[CoordinateAngleDistance, CoordinateAngleDistance, CoordinateAngleDistance, CoordinateAngleDistance]:
#     while len(ld_low_angle_quarter) > 1:
#         ld_low_angle_quarter, _ = split_in_half_by_distance(ld_low_angle_quarter)
#         ld_low_angle_quarter, _ = split_in_half_by_angle(ld_low_angle_quarter)
#         ld_low_angle_quarter = list(set(ld_low_angle_quarter))
#     while len(hd_low_angle_quarter) > 1:
#         _, hd_low_angle_quarter = split_in_half_by_distance(hd_low_angle_quarter)
#         hd_low_angle_quarter, _ = split_in_half_by_angle(hd_low_angle_quarter)
#         hd_low_angle_quarter = list(set(hd_low_angle_quarter))
#     while len(hd_high_angle_quarter) > 1:
#         _, hd_high_angle_quarter = split_in_half_by_distance(hd_high_angle_quarter)
#         _, hd_high_angle_quarter = split_in_half_by_angle(hd_high_angle_quarter)
#         hd_high_angle_quarter = list(set(hd_high_angle_quarter))
#     while len(ld_high_angle_quarter) > 1:
#         ld_high_angle_quarter, _ = split_in_half_by_distance(ld_high_angle_quarter)
#         _, ld_high_angle_quarter = split_in_half_by_angle(ld_high_angle_quarter)
#         ld_high_angle_quarter = list(set(ld_high_angle_quarter))
#
#     return ld_low_angle_quarter[0], hd_low_angle_quarter[0], hd_high_angle_quarter[0], ld_high_angle_quarter[0]
#
#
# def get_cads_corners(cads, segment_type: SegmentType) -> RadialSectionRingIntersection or None:
#     if segment_type == SegmentType.INNER or segment_type == SegmentType.OUTER:
#
#         return get_inner_or_outer_segment_corners(cads)
#
#     elif segment_type == SegmentType.TRIPLE or segment_type == SegmentType.DOUBLE:
#
#         return get_triple_or_double_segment_corners(cads)
#
#     elif segment_type == SegmentType.BULL or segment_type == SegmentType.BULLS_EYE:
#
#         return None
#
