from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import cv2 as cv

from data_structures.coordinates import Coordinate2d
from data_structures.score_segments import ScoreSegment, SegmentType
from operations_on.angles import get_angle_range
from operations_on.coordinates_and_angles import get_endpoint_via, get_endpoint_range, get_endpoint_via_with_floats
from operations_on.lists import flatten


@dataclass
class __DartboardDefinition:
    """This circled board calculation holds predefined values to represent a dartboard
    It is used to un-distort a dartboard image with OpenCVs camera calibration functions
    """
    # --- Dartboard definitions
    # the dimensions can be found here: https://www.dimensions.com/element/dartboard
    # the unit is millimeter [mm]
    board_diameter = 451
    # Triple and Double segments are defined as 8mm in width
    # My own dartboard seems to have a width of 9mm, it's an old Winmau Blade 4. Is that cheating?
    # Follow me for more tips how to get a better average!
    # In the end it doesn't matter for the purpose of this calculation, which is to segment a dartboard img
    triple_double_segment_width = 8
    # The Bullseye segment is 12.7mm in outer diameter which gives us
    bulls_eye_radius = 12.7 / 2
    bulls_eye_center_floats = (board_diameter / 2, board_diameter / 2)
    # The Bull segment is 32mm in outer diameter which gives us
    bull_radius = 32 / 2
    # From Bulls-eye to the outside of the triple segments we have a definition of 107mm
    triple_outer_radius = 107
    # For inner radius the spider-wire has to be taken into consideration
    triple_inner_radius = triple_outer_radius - triple_double_segment_width
    # From Bulls-eye to the outside of the double segments we have a definition of 170mm
    double_outer_radius = 170
    double_inner_radius = double_outer_radius - triple_double_segment_width
    # The radial sections are angled
    # Here, radians are used for the calculations
    # 0 equals to the middle of radial section for letter "six"
    # PI equals to the middle of radial section for letter "eleven"
    radial_section_angle_span = 2 * np.pi / 20
    # Each section is rotated by half of the angle width
    # Here the concrete corner definitions begin
    # The corners are defined clockwise from angle 0
    # The first radial section transition is from number six to ten
    # Angle 0 is the middle of the radial section of six
    # The minus is here to give a smoother for loop
    transition_angle_start = -(radial_section_angle_span / 2)
    # In addition to the sizes of segments, the number ring has also a defined order
    # For ease of use with radians, the start is at number 10, which is the first complete section clockwise
    # starting at radian 0
    letter_sequence_clockwise = [10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6]

    # --- Project definitions
    # A scale is applied to the Millimeter source calculation to get precise pixel coordinates for board segment edges
    # The scale is uneven so the bulls_eye_mid can be a Tuple[int, int] instead of a Tuple[float, float]
    _scale = 3
    board_diameter_scaled = board_diameter * _scale
    bulls_eye_radius_scaled = bulls_eye_radius * _scale
    bull_radius_scaled = bull_radius * _scale
    triple_inner_radius_scaled = triple_inner_radius * _scale
    triple_outer_radius_scaled = triple_outer_radius * _scale
    double_inner_radius_scaled = double_inner_radius * _scale
    double_outer_radius_scaled = double_outer_radius * _scale
    bulls_eye_center_coordinate_scaled = Coordinate2d(
        np.uint32(board_diameter_scaled / 2),
        np.uint32(board_diameter_scaled / 2)
    )
    # In the process of applying the warped segments onto the live image, the segments have to have more than 4 points
    # This number of points is generated for the ring parts of a section
    coordinates_per_radial_angled_section = 20
    coordinates_per_inner_outer_section_transition = 100
    coordinates_per_triple_double_section_transition = 3

    length_to_bull = bull_radius / double_outer_radius
    length_to_inner_triple = triple_inner_radius / double_outer_radius
    length_to_outer_triple = triple_outer_radius / double_outer_radius
    length_to_double_inner = double_inner_radius / double_outer_radius


def get_real_world_coordinates() -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preset coordinates for a circled dartboard, used for camera calibration"""
    # The first coordinate is the board center.
    # Then other coordinates of corners are added.
    # The complete board corner definition follows the format of 20 x 4 corner points per radial section transition
    # 1. corner of bull to inner
    # 2. corner of inner to triple
    # 3. corner of triple to outer
    # 4. corner of outer to double
    # 5. corner of double to board
    board_corners = [__DartboardDefinition.bulls_eye_center_floats]
    angle = __DartboardDefinition.transition_angle_start
    for _ in range(20):
        angle = angle + __DartboardDefinition.radial_section_angle_span
        # board_corners.append(
        #     get_endpoint_via_with_floats(
        #         __DartboardDefinition.bulls_eye_center_floats,
        #         angle,
        #         __DartboardDefinition.bull_radius
        #     )
        # )
        board_corners.append(
            get_endpoint_via_with_floats(
                __DartboardDefinition.bulls_eye_center_floats,
                angle,
                __DartboardDefinition.triple_inner_radius
            )
        )
        board_corners.append(
            get_endpoint_via_with_floats(
                __DartboardDefinition.bulls_eye_center_floats,
                angle,
                __DartboardDefinition.triple_outer_radius
            )
        )
        board_corners.append(
            get_endpoint_via_with_floats(
                __DartboardDefinition.bulls_eye_center_floats,
                angle,
                __DartboardDefinition.double_inner_radius
            )
        )
        board_corners.append(
            get_endpoint_via_with_floats(
                __DartboardDefinition.bulls_eye_center_floats,
                angle,
                __DartboardDefinition.double_outer_radius
            )
        )

    # reduce board_corners to a numpy array
    board_corners = np.asarray([(corner[0], corner[1], 0) for corner in board_corners], dtype=np.float32)

    image_dims = (__DartboardDefinition.board_diameter, __DartboardDefinition.board_diameter)

    return board_corners, image_dims



def get_real_world_coordinates() -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preset coordinates for a circled dartboard, used for camera calibration"""
    # The first coordinate is the board center.
    # Then other coordinates of corners are added.
    # The complete board corner definition follows the format of 20 x 4 corner points per radial section transition
    # 1. corner of bull to inner
    # 2. corner of inner to triple
    # 3. corner of triple to outer
    # 4. corner of outer to double
    # 5. corner of double to board
    board_corners = [__DartboardDefinition.bulls_eye_center_floats]
    angle = __DartboardDefinition.transition_angle_start
    for _ in range(20):
        angle = angle + __DartboardDefinition.radial_section_angle_span
        # board_corners.append(
        #     get_endpoint_via_with_floats(
        #         __DartboardDefinition.bulls_eye_center_floats,
        #         angle,
        #         __DartboardDefinition.bull_radius
        #     )
        # )
        board_corners.append(
            get_endpoint_via_with_floats(
                __DartboardDefinition.bulls_eye_center_floats,
                angle,
                __DartboardDefinition.triple_inner_radius
            )
        )
        board_corners.append(
            get_endpoint_via_with_floats(
                __DartboardDefinition.bulls_eye_center_floats,
                angle,
                __DartboardDefinition.triple_outer_radius
            )
        )
        board_corners.append(
            get_endpoint_via_with_floats(
                __DartboardDefinition.bulls_eye_center_floats,
                angle,
                __DartboardDefinition.double_inner_radius
            )
        )
        board_corners.append(
            get_endpoint_via_with_floats(
                __DartboardDefinition.bulls_eye_center_floats,
                angle,
                __DartboardDefinition.double_outer_radius
            )
        )

    # reduce board_corners to a numpy array
    board_corners = np.asarray([(corner[0], corner[1], 0) for corner in board_corners], dtype=np.float32)

    image_dims = (__DartboardDefinition.board_diameter, __DartboardDefinition.board_diameter)

    return board_corners, image_dims


def get_preset_board_coordinates() -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preset coordinates for a scaled circled dartboard representation on an image"""
    # The complete board corner definition follows the format of 20 x 5 corner points per radial section transition
    # 1. corner of bull to inner
    # 2. corner of inner to triple
    # 3. corner of triple to outer
    # 4. corner of outer to double
    # 5. corner of double to board
    board_corners = []
    angle = __DartboardDefinition.transition_angle_start
    for _ in range(20):
        angle = angle + __DartboardDefinition.radial_section_angle_span
        board_corners.append(
            get_endpoint_via(
                __DartboardDefinition.bulls_eye_center_coordinate_scaled,
                angle,
                __DartboardDefinition.bull_radius_scaled
            )
        )
        board_corners.append(
            get_endpoint_via(
                __DartboardDefinition.bulls_eye_center_coordinate_scaled,
                angle,
                __DartboardDefinition.triple_inner_radius_scaled
            )
        )
        board_corners.append(
            get_endpoint_via(
                __DartboardDefinition.bulls_eye_center_coordinate_scaled,
                angle,
                __DartboardDefinition.triple_outer_radius_scaled
            )
        )
        board_corners.append(
            get_endpoint_via(
                __DartboardDefinition.bulls_eye_center_coordinate_scaled,
                angle,
                __DartboardDefinition.double_inner_radius_scaled
            )
        )
        board_corners.append(
            get_endpoint_via(
                __DartboardDefinition.bulls_eye_center_coordinate_scaled,
                angle,
                __DartboardDefinition.double_outer_radius_scaled
            )
        )

    # reduce board_corners to a numpy array
    board_corners = np.asarray([(corner.x, corner.y) for corner in board_corners], dtype=np.int32)

    image_dims = (__DartboardDefinition.board_diameter_scaled, __DartboardDefinition.board_diameter_scaled)

    return board_corners, image_dims


def __get_inner_segment_contour(angle_low, angle_high) -> np.ndarray:
    linspace_distance = np.linspace(
        __DartboardDefinition.bull_radius_scaled,
        __DartboardDefinition.triple_inner_radius_scaled,
        num=__DartboardDefinition.coordinates_per_inner_outer_section_transition
    )
    all_coordinates_np = __get_contour_from_linspaces(
        angle_high,
        angle_low,
        linspace_distance,
        __DartboardDefinition.coordinates_per_inner_outer_section_transition
    )

    return all_coordinates_np


def __get_triple_segment_contour(angle_low, angle_high) -> np.ndarray:
    linspace_distance = np.linspace(
        __DartboardDefinition.triple_inner_radius_scaled,
        __DartboardDefinition.triple_outer_radius_scaled,
        num=__DartboardDefinition.coordinates_per_triple_double_section_transition
    )

    all_coordinates_np = __get_contour_from_linspaces(
        angle_high,
        angle_low,
        linspace_distance,
        __DartboardDefinition.coordinates_per_triple_double_section_transition
    )

    return all_coordinates_np


def __get_outer_segment_contour(angle_low, angle_high) -> np.ndarray:
    linspace_distance = np.linspace(
        __DartboardDefinition.triple_outer_radius_scaled,
        __DartboardDefinition.double_inner_radius_scaled,
        num=__DartboardDefinition.coordinates_per_inner_outer_section_transition
    )

    all_coordinates_np = __get_contour_from_linspaces(
        angle_high,
        angle_low,
        linspace_distance,
        __DartboardDefinition.coordinates_per_inner_outer_section_transition
    )

    return all_coordinates_np


def __get_double_segment_contour(angle_low, angle_high) -> np.ndarray:
    linspace_distance = np.linspace(
        __DartboardDefinition.double_inner_radius_scaled,
        __DartboardDefinition.double_outer_radius_scaled,
        num=__DartboardDefinition.coordinates_per_triple_double_section_transition
    )

    all_coordinates_np = __get_contour_from_linspaces(
        angle_high,
        angle_low,
        linspace_distance,
        __DartboardDefinition.coordinates_per_triple_double_section_transition
    )

    return all_coordinates_np


def __get_contour_from_linspaces(angle_high, angle_low, linspace_distance, num_distances):
    min_distance = min(linspace_distance)
    max_distance = max(linspace_distance)
    linspace_angle = get_angle_range(
        angle_low,
        angle_high,
        num=__DartboardDefinition.coordinates_per_radial_angled_section
    )
    edge_low_angle = get_endpoint_range(
        __DartboardDefinition.bulls_eye_center_coordinate_scaled,
        [angle_low] * num_distances,
        linspace_distance
    )
    edge_radial_high_distance = get_endpoint_range(
        __DartboardDefinition.bulls_eye_center_coordinate_scaled,
        linspace_angle,
        [max_distance] * __DartboardDefinition.coordinates_per_radial_angled_section
    )
    edge_high_angle = get_endpoint_range(
        __DartboardDefinition.bulls_eye_center_coordinate_scaled,
        [angle_high] * num_distances,
        linspace_distance
    )
    edge_high_angle.reverse()
    edge_radial_low_distance = get_endpoint_range(
        __DartboardDefinition.bulls_eye_center_coordinate_scaled,
        linspace_angle,
        [min_distance] * __DartboardDefinition.coordinates_per_radial_angled_section
    )
    edge_radial_low_distance.reverse()
    all_coordinates = flatten(
        [edge_low_angle, edge_radial_high_distance, edge_high_angle, edge_radial_low_distance])
    all_coordinates_np = np.asarray([(cad.coordinate.x, cad.coordinate.y) for cad in all_coordinates])
    return all_coordinates_np


def warp_coordinate(x, y, homography_matrix: np.ndarray):
    """

    Args:
        x: Part of a coordinate
        y: Part of a coordinate
        homography_matrix: 3x3 numpy array

    Returns: warped coordinates

    References: https://stackoverflow.com/a/42410728

    """
    p = np.array((x, y, 1)).reshape((3, 1))
    temp_p = homography_matrix.dot(p)
    sum = np.sum(temp_p, 1)
    px = int(round(sum[0] / sum[2]))
    py = int(round(sum[1] / sum[2]))

    return px, py


def get_preset_board_segments(homography_matrix: np.ndarray, img) -> List[ScoreSegment]:
    """Preset segment borders, used to calculate the warped segment borders with a homography matrix

    Args:
        homography_matrix (np.ndarray): Homography matrix from circled img to live img
    """
    # The first calculated section not a section, but the bulls-eye segment
    # followed by the outer contour of the bull segment
    # Then all the radial sections are appended to the List
    # The first calculated section is for the number ten, order is from inner to double
    bulls_eye_coordinates = []
    bull_coordinates = []
    angle = __DartboardDefinition.transition_angle_start
    for _ in range(20):
        angle = angle + __DartboardDefinition.radial_section_angle_span
        bulls_eye_coordinates.append(
            get_endpoint_via(
                __DartboardDefinition.bulls_eye_center_coordinate_scaled,
                angle,
                __DartboardDefinition.bulls_eye_radius_scaled
            )
        )
        bull_coordinates.append(
            get_endpoint_via(
                __DartboardDefinition.bulls_eye_center_coordinate_scaled,
                angle,
                __DartboardDefinition.bull_radius_scaled
            )
        )
    score_segments: List[ScoreSegment] = []
    bulls_eye_coordinates_np = np.asarray([(c.x, c.y) for c in bulls_eye_coordinates])
    score_segment_bulls_eye = ScoreSegment(bulls_eye_coordinates_np, SegmentType.BULLS_EYE)
    score_segments.append(score_segment_bulls_eye)
    bull_coordinates_np = np.asarray([(c.x, c.y) for c in bull_coordinates])
    score_segment_bull = ScoreSegment(bull_coordinates_np, SegmentType.BULLS_EYE)
    score_segments.append(score_segment_bull)

    angle_low = __DartboardDefinition.transition_angle_start
    angle_high = __DartboardDefinition.transition_angle_start + __DartboardDefinition.radial_section_angle_span
    for i in range(20):
        letter = __DartboardDefinition.letter_sequence_clockwise[i]
        angle_low = angle_low + __DartboardDefinition.radial_section_angle_span
        angle_high = angle_high + __DartboardDefinition.radial_section_angle_span

        inner_segment = __get_inner_segment_contour(angle_low, angle_high)
        score_segment_inner = ScoreSegment(inner_segment, letter)
        score_segments.append(score_segment_inner)

        triple_segment = __get_triple_segment_contour(angle_low, angle_high)
        score_segment_triple = ScoreSegment(triple_segment, letter)
        score_segments.append(score_segment_triple)

        outer_segment = __get_outer_segment_contour(angle_low, angle_high)
        score_segment_outer = ScoreSegment(outer_segment, letter)
        score_segments.append(score_segment_outer)

        double_segment = __get_double_segment_contour(angle_low, angle_high)
        score_segment_double = ScoreSegment(double_segment, letter)
        score_segments.append(score_segment_double)

    score_segment_img = img.copy()
    for score_segment in score_segments:
        cv.drawContours(score_segment_img, [score_segment.contour], -1, (255, 255, 0), 1)

    # now we can apply the perspective transform on all segment contours
    for score_segment in score_segments:
        contour = score_segment.contour
        warped_contour = np.asarray([warp_coordinate(c[0], c[1], homography_matrix) for c in contour])
        score_segment.contour = warped_contour

    return score_segments, score_segment_img
