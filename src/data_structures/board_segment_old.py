from abc import ABC
from dataclasses import dataclass, field
from typing import List

import cv2 as cv

import numpy as np

from data_structures.interpolation import EdgeValues, AngleDistanceInterpolation
from operations_on.angles import get_angle_range, get_angle_diff, is_angle_in_range, get_min_angle, get_max_angle
from operations_on.contours import filter_min_size_contour, filter_max_size_contour, filter_center_inside_contour, \
    filter_contour_solidity
from operations_on.ellipses import get_fit_ellipse
from operations_on.coordinates import get_distance, get_mid_coordinate, get_points_as_nparray, \
    get_interpolation
from operations_on.coordinates_and_angles import get_angle, get_endpoint_range, get_cads
from src.data_structures.coordinates import Coordinate2d

from src.data_structures.score_segments import SegmentType
from data_structures.score_segment_old import ScoreSegmentOld, BullOrBullseyeScoreSegment, NumberedScoreSegment


@dataclass
class BoardSegment:
    contour: np.ndarray  # extracted from hsv filters
    preprocessed_contour: ScoreSegmentOld = field(init=False)  # preprocessed contour
    aligned_contour: ScoreSegmentOld = field(init=False)  # further processing by aligning with other contours
    score_border: ScoreSegmentOld = field(init=False)  # used for score calculation
    cp: Coordinate2d = field(init=False)  # center point of the contour, pixel_x and pixel_y
    isolated_cp: Coordinate2d = field(init=False)  # cp but in transformed coordinates
    isolated_distance_from_board_center: float = field(init=False)  # distance to center but transformed coordinates
    segment_type: SegmentType = field(init=False)
    extrapolated_end_point: Coordinate2d = field(init=False)  # used for inner and other field association
    associated_number: int = field(init=False)

    def __post_init__(self):
        self.cp = self.get_center_of_contour()

    def set_isolated_cp(self, point: Coordinate2d):
        self.isolated_cp = point

    def calculate_isolated_distance(self, point: Coordinate2d):
        try:
            self.isolated_distance_from_board_center = round(get_distance(self.isolated_cp, point), 2)
        except AttributeError as ae:
            x = 0

    def get_center_of_contour(self):
        m = cv.moments(self.contour)
        try:
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
        except ZeroDivisionError as zde:
            cx = self.contour[0][0][0]
            cy = self.contour[0][0][1]
        return Coordinate2d(cx, cy)

    def get_contour_distance_to(self, point: Coordinate2d):
        # returns negative number if point is outside the contour
        # returns 0 if point is on the contour
        # returns positive number if point is inside the contour
        return cv.pointPolygonTest(self.contour, (point.x, point.y), measureDist=True)

    def filter_center_inside_contour(self):
        dist = cv.pointPolygonTest(self.contour, (self.cp.x, self.cp.y), measureDist=False)
        return dist >= 0  # contour center is inside the contour

    def filter_max_size_contour(self):
        # not used
        return cv.contourArea(self.contour) < 10000

    def filter_min_size_contour(self):
        # not used
        return cv.contourArea(self.contour) > 150

    def filter_contour_solidity(self):
        # not used
        # Solidity is the ratio of contour area to its convex hull area
        area = cv.contourArea(self.contour)
        hull = cv.convexHull(self.contour)
        hull_area = cv.contourArea(hull)
        solidity = float(area) / hull_area
        return solidity > 0.7

    def is_probably_a_board_field(self, board_segment_min_size, board_segment_max_size):
        return (
                filter_min_size_contour(self.contour, board_segment_min_size) and
                filter_max_size_contour(self.contour, board_segment_max_size) and
                filter_center_inside_contour(self.contour, self.cp) and
                filter_contour_solidity(self.contour)
        )

    def get_approximated_contour(self):
        # not used
        arc_length = 5  # maximum distance from contour to approximated contour
        epsilon = arc_length / 1000 * cv.arcLength(self.contour, True)
        approx = cv.approxPolyDP(self.contour, epsilon, True)
        return approx

    def get_shape_edge_towards(self, end_point: Coordinate2d) -> Coordinate2d:
        # approximated contour rim point
        search_iterations = 6
        for _ in range(search_iterations):
            p_range = np.linspace((self.cp.x, self.cp.y), (end_point.x, end_point.y), 4, dtype=int)
            p_range = [Coordinate2d(int(x), int(y)) for x, y in p_range]
            distances = [
                (self.get_contour_distance_to(p_range[0]), p_range[0]),
                (self.get_contour_distance_to(p_range[1]), p_range[1]),
                (self.get_contour_distance_to(p_range[2]), p_range[2]),
                (self.get_contour_distance_to(p_range[3]), p_range[3])
            ]
            distances_positive = [dist for dist in distances if dist[0] >= 0]
            distances_negative = [dist for dist in distances if dist[0] < 0]
            distances_positive = sorted(distances_positive, key=lambda x: x[0])
            distances_negative = sorted(distances_negative, key=lambda x: x[0], reverse=True)

            p1 = distances_negative[0][1]
            p2 = distances_positive[0][1]

            if p1 == p2:
                break

        return get_mid_coordinate(p1, p2)

    def get_contour_points_angles_distances(self, board_cp: Coordinate2d) -> List[Coordinate2d]:
        contour_points = [Coordinate2d(point[0][0], point[0][1]) for point in self.contour]
        for p in contour_points:
            p.distance_to_board_center = get_distance(p, board_cp)
            p.angle_to_board_center = get_angle(p, board_cp)
        return contour_points


@dataclass
class SpecializedBoardSegment(BoardSegment):
    score_multiplication = 1

    def __init__(self, board_field: BoardSegment):
        super().__init__(board_field.contour)


@dataclass
class BullBullsEyeSegment(SpecializedBoardSegment):
    contour_points: List[Coordinate2d] = field(default_factory=list)
    preprocessed_contour: BullOrBullseyeScoreSegment = field(default=BullOrBullseyeScoreSegment)
    score_border: BullOrBullseyeScoreSegment = field(default_factory=BullOrBullseyeScoreSegment)

    def add_calculated_border_point(self, point: Coordinate2d):
        self.score_border.add_point(point)

    def calculate_point_angles_and_distances(self, board_cp: Coordinate2d):
        self.contour_points = [Coordinate2d(point[0][0], point[0][1]) for point in self.contour]
        for p in self.contour_points:
            p.distance_to_board_center = get_distance(p, board_cp)
            p.angle_to_board_center = get_angle(p, board_cp)

    def calculate_interpolation(self, cam_resolution, board_cp: Coordinate2d):
        angles, smoothed_distances = get_interpolation(self.contour_points)
        smoothed_points = get_endpoint_range(board_cp, angles, smoothed_distances)
        smoothed_points = get_points_as_nparray(smoothed_points)
        ellipse_points = get_fit_ellipse(smoothed_points, cam_resolution)
        ellipse_points = get_cads(ellipse_points, board_cp)
        smoothed_ellipse_angles, smoothed_ellipse_distances = get_interpolation(ellipse_points)
        mean_distances = np.mean([smoothed_distances, smoothed_ellipse_distances], axis=0)
        a_d_i_e_1_4 = AngleDistanceInterpolation(angles, mean_distances)

        return a_d_i_e_1_4


@dataclass
class BullsEyeSegment(BullBullsEyeSegment):
    associated_number = 50


@dataclass
class BullSegment(BullBullsEyeSegment):
    associated_number = 25


@dataclass
class NumberedSegment(SpecializedBoardSegment, ABC):
    e_1_4: EdgeValues = field(default_factory=list)
    e_2_3: EdgeValues = field(default_factory=list)
    contour_points: List[Coordinate2d] = field(default_factory=list)
    # preprocessed_contour: NumberedScoreSegment = field(default_factory=NumberedScoreSegment)
    # aligned_contour: NumberedScoreSegment = field(default_factory=NumberedScoreSegment)
    score_border: NumberedScoreSegment = field(default_factory=NumberedScoreSegment)

    def calculate_point_angles_and_distances(self, board_cp: Coordinate2d):
        self.contour_points = [Coordinate2d(point[0][0], point[0][1]) for point in self.contour]
        for p in self.contour_points:
            p.distance_to_board_center = get_distance(p, board_cp)
            p.angle_to_board_center = get_angle(p, board_cp)

    def _get_edge_values(self, e_1_4_dist_cutoff: float, e_2_3_dist_cutoff: float, e_1_4_angle_cutoff: float,
                         e_2_3_angle_cutoff: float):

        min_angle = min(self.contour_points, key=lambda p: p.angle).angle_to_board_center
        max_angle = max(self.contour_points, key=lambda p: p.angle).angle_to_board_center
        if min_angle < 1 and max_angle > 5:  # radian max "overflow"
            min_angle = min(
                [p for p in self.contour_points if p.angle_to_board_center > 4],
                key=lambda p: p.angle
            ).angle_to_board_center
            max_angle = max(
                [p for p in self.contour_points if p.angle_to_board_center < 2],
                key=lambda p: p.angle
            ).angle_to_board_center

        angle_diff = get_angle_diff(max_angle, min_angle)
        angle_linspace = get_angle_range(min_angle, max_angle, num=6)

        e_1_4_points = []  # edge with less distance to board center
        e_2_3_points = []  # edge with more distance to board center

        angle_part_1 = [
            p for p in self.contour_points if
            is_angle_in_range(p.angle_to_board_center, angle_linspace[0], angle_linspace[1])
        ]
        angle_part_2 = [
            p for p in self.contour_points if
            is_angle_in_range(p.angle_to_board_center, angle_linspace[1], angle_linspace[2])
        ]
        angle_part_3 = [
            p for p in self.contour_points if
            is_angle_in_range(p.angle_to_board_center, angle_linspace[2], angle_linspace[3])
        ]
        angle_part_4 = [
            p for p in self.contour_points if
            is_angle_in_range(p.angle_to_board_center, angle_linspace[3], angle_linspace[4])
        ]
        angle_part_5 = [
            p for p in self.contour_points if
            is_angle_in_range(p.angle_to_board_center, angle_linspace[4], angle_linspace[5])
        ]

        for angle_part in [angle_part_1, angle_part_2, angle_part_3, angle_part_4, angle_part_5]:
            min_distance = min(
                angle_part,
                key=lambda p: p.distance,
                default=Coordinate2d(0, 0)
            ).distance_to_board_center

            max_distance = max(
                angle_part,
                key=lambda p: p.distance,
                default=Coordinate2d(0, 0)
            ).distance_to_board_center

            distance_diff = max_distance - min_distance
            e_1_4_points += [
                p for p in angle_part if
                p.distance_to_board_center < min_distance + distance_diff * e_1_4_dist_cutoff
            ]
            e_2_3_points += [
                p for p in angle_part if
                p.distance_to_board_center > max_distance - distance_diff * e_2_3_dist_cutoff
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

        self.e_1_4 = EdgeValues(e_1_4_points, c_1_angle, c_4_angle)
        self.e_2_3 = EdgeValues(e_2_3_points, c_2_angle, c_3_angle)


@dataclass
class InnerSegment(NumberedSegment):
    segment_type = SegmentType.INNER

    def get_edge_values(self):
        self._get_edge_values(
            0.01,
            0.1,
            0.4,
            0.15
        )


@dataclass
class TripleSegment(NumberedSegment):
    segment_type = SegmentType.TRIPLE
    score_multiplication = 3

    def get_edge_values(self):
        self._get_edge_values(
            0.5,
            0.5,
            0.1,
            0.1
        )


@dataclass
class OuterSegment(NumberedSegment):
    segment_type = SegmentType.OUTER

    def get_edge_values(self):
        self._get_edge_values(
            0.1,
            0.2,
            0.1,
            0.1
        )


@dataclass
class DoubleSegment(NumberedSegment):
    segment_type = SegmentType.DOUBLE
    score_multiplication = 2

    def get_edge_values(self):
        self._get_edge_values(
            0.5,
            0.5,
            0.1,
            0.1
        )


@dataclass
class AssociatedBoardSegments:
    associated_number: int = field(init=False)
    inner_segment: InnerSegment
    triple_segment: TripleSegment
    outer_segment: OuterSegment
    double_segment: DoubleSegment

    def __init__(self, number_segments: [BoardSegment]):
        self.inner_segment = InnerSegment(number_segments[0].contour)
        self.triple_segment = TripleSegment(number_segments[1].contour)
        self.outer_segment = OuterSegment(number_segments[2].contour)
        self.double_segment = DoubleSegment(number_segments[3].contour)

    def set_number(self, associated_number: int):
        self.associated_number = associated_number
        self.inner_segment.associated_number = associated_number
        self.triple_segment.associated_number = associated_number
        self.outer_segment.associated_number = associated_number
        self.double_segment.associated_number = associated_number

    def get_sections(self) -> List[SpecializedBoardSegment]:
        return [
            self.inner_segment,
            self.triple_segment,
            self.outer_segment,
            self.double_segment
        ]
