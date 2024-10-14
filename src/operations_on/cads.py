from typing import List

import numpy as np

from data_structures.board_segment import BoardSegment
from data_structures.coordinates import Coordinate2d
from data_structures.coordinates_extended import CoordinateAngleDistance, CoordinateGroup
from operations_on.angles import get_angle_diff
from operations_on.angles_and_distances import get_interpolated_coordinates_with_smoothed_angles, \
    get_angles_shifted_to_safe_range
from operations_on.coordinates import get_interpolation, get_points_as_nparray
from operations_on.coordinates_and_angles import get_cads, get_endpoint_range
from operations_on.ellipses import get_fit_ellipse


def smooth_transition_point_groups(board_center_coordinate: Coordinate2d, radial_section_transitions: List[List[CoordinateAngleDistance]]) -> List[List[CoordinateAngleDistance]]:
    smoothed_transition_point_groups = []
    for coordinate_group in radial_section_transitions:
        coordinate_group.sort(key=lambda x: x.distance)

        angles = [cad.angle for cad in coordinate_group]
        distances = [cad.distance for cad in coordinate_group]

        # angle_diffs = get_angle_diffs(angles)
        # angle_diff_cutoff = (2 * np.pi) / 400
        # abs_angle_diffs = np.asarray([abs(diff) for diff in angle_diffs])
        # bad_angles_indices = np.argwhere(abs_angle_diffs > angle_diff_cutoff)
        # if bad_angles_indices.size > 0:
        #     bad_angles_indices = bad_angles_indices + 1  # np.argwhere returns one i earlier then the actual bad i
        #     bad_angles_indices = [i[0] for i in bad_angles_indices]
        #     # remove bad angles that are too far from the angles around them
        #     [angles.pop(i) for i in bad_angles_indices if i < len(angles)]
        #     [distances.pop(i) for i in bad_angles_indices if i < len(distances)]

        new_coordinates: List[CoordinateAngleDistance] = get_interpolated_coordinates_with_smoothed_angles(angles, distances, board_center_coordinate, 500, 105)
        smoothed_transition_point_groups.append(new_coordinates)

    return smoothed_transition_point_groups


def get_angle_diffs(angles):
    angles_shifted, _ = get_angles_shifted_to_safe_range(angles)

    return np.diff(angles_shifted).astype(np.float32)


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


def get_smoothed_bull_or_bulls_eye_contour(img_width, img_height, segment: BoardSegment) -> np.ndarray:
    angles, smoothed_distances = get_interpolation(segment.contour_cads)
    smoothed_points = get_endpoint_range(segment.center_cad.coordinate, angles, smoothed_distances, safe_angle_and_distance=False)
    smoothed_points = get_points_as_nparray(smoothed_points)
    ellipse_points = get_fit_ellipse(smoothed_points, img_width, img_height)
    ellipse_points = get_cads(ellipse_points, segment.center_cad.coordinate)
    smoothed_ellipse_angles, smoothed_ellipse_distances = get_interpolation(ellipse_points)
    # mean_distances = np.mean([smoothed_distances, smoothed_ellipse_distances], axis=0)

    smoothed_contour = get_endpoint_range(segment.center_cad.coordinate, smoothed_ellipse_angles, smoothed_ellipse_distances, safe_angle_and_distance=False)
    smoothed_contour = np.asarray([(c.x, c.y) for c in smoothed_contour], dtype=np.int32)

    return smoothed_contour


def get_cads_as_contour(cads: List[CoordinateAngleDistance]) -> np.ndarray:
    contour = np.asarray([(c.coordinate.x, c.coordinate.y) for c in cads], dtype=np.int32)

    return contour


def get_cad_with_adjacent_angle(cads: List[CoordinateAngleDistance], angle: float) -> CoordinateAngleDistance:
    angle_diffs = [get_angle_diff(cad.angle, angle) for cad in cads]
    best_match_index = np.argmin(angle_diffs)
    match_cad = cads[best_match_index]

    return match_cad


def get_adjacent_angle_cad(cad_original: CoordinateAngleDistance, cads_destination: List[CoordinateAngleDistance]):
    angle_diffs_to_cad = np.asarray([np.abs(cad.angle - cad_original.angle) for cad in cads_destination])
    i_closest_angle = np.argmin(angle_diffs_to_cad)
    closest_cad_by_angle = cads_destination[i_closest_angle]
    return closest_cad_by_angle
