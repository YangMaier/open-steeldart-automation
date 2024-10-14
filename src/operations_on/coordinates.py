from typing import List

import numpy as np

from data_structures.coordinates import Coordinate2dNormalized, Coordinate2d
from data_structures.coordinates_extended import CoordinateAngleDistance
from operations_on.lists import mask_first_occurrence_with_unique, get_extended_angles, get_extended_list, \
    smooth_savgol_filter, mask_angles_in_radian_range


def line_line_intersection(p1: Coordinate2d, p2: Coordinate2d, p3: Coordinate2d, p4: Coordinate2d):
    """
    Calculate the point of intersection between two lines defined by points p1, p2 and p3, p4.

    Parameters:
    - p1: Point object representing the first point of the first line
    - p2: Point object representing the second point of the first line
    - p3: Point object representing the first point of the second line
    - p4: Point object representing the second point of the second line

    Returns:
    - Point object representing the point of intersection of the two lines

    References:
    - https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/
    """
    # Line AB represented as a1x + b1y = c1
    a1 = p2.y - p1.y
    b1 = p1.x - p2.x
    c1 = a1 * p1.x + b1 * p1.y

    # Line CD represented as a2x + b2y = c2
    a2 = p4.y - p3.y
    b2 = p3.x - p4.x
    c2 = a2 * p3.x + b2 * p3.y

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        # The lines are parallel. This is simplified
        # by returning a pair of FLT_MAX

        return Coordinate2d(10 ** 9, 10 ** 9)

    else:
        x = int((b2 * c1 - b1 * c2) / determinant)
        y = int((a1 * c2 - a2 * c1) / determinant)

        return Coordinate2d(x, y)


def get_distance(point1: Coordinate2d, point2: Coordinate2d) -> np.float32:
    distance = np.linalg.norm(np.array((point1.x, point1.y)) - np.array((point2.x, point2.y)))
    distance = np.float32(distance)
    return distance


def is_point_in_tolerance(point1: Coordinate2d, point2: Coordinate2d, tolerance_distance: float = 3.0) -> bool:

    return get_distance(point1, point2) < tolerance_distance


def get_mid_coordinate(point1: Coordinate2d, point2: Coordinate2d) -> Coordinate2d:

    return Coordinate2d(int((point1.x + point2.x) * 0.5), int((point1.y + point2.y) * 0.5))


def get_coordinate_linspace(point1: Coordinate2d, point2: Coordinate2d, quantity=10) -> [Coordinate2d]:
    x_range = np.linspace(point1.x, point2.x, quantity).astype(np.uint32)
    y_range = np.linspace(point1.y, point2.y, quantity).astype(np.uint32)

    coordinate_range = [Coordinate2d(x, y) for x, y in zip(x_range, y_range)]

    return coordinate_range


def get_line_extension_end_point(start_point: Coordinate2d, end_point: Coordinate2d, length) -> Coordinate2d:
    # visualize a line from point a to b, then extend the line by times length,
    # the end point of that line is returned
    end_x = np.int32(start_point.x + (end_point.x - start_point.x) * length)
    end_y = np.int32(start_point.y + (end_point.y - start_point.y) * length)

    return Coordinate2d(end_x, end_y)


def get_points_as_nparray(points: [Coordinate2d]) -> np.ndarray:

    return np.array([[point.x, point.y] for point in points])


def get_interpolation(cads: List[CoordinateAngleDistance], minimize=False, maximize=False):
    cads = sorted(cads, key=lambda p: p.angle)
    angles = [p.angle for p in cads]
    # It's probably a good idea to drop angle duplicates
    unique_mask = mask_first_occurrence_with_unique(angles)
    distances = [p.distance for p in cads]
    angles = list(np.array(angles)[~unique_mask])
    distances = list(np.array(distances)[~unique_mask])

    # Near radian max and min is an open interval, we want to close it
    angles = get_extended_angles(angles)
    distances = get_extended_list(distances)

    radians_max = 2 * np.pi
    # a fixed interval for all angles makes it easier to smooth the distances
    angle_linspace = np.linspace(0 - radians_max * 0.1, radians_max * 1.1, 2000)
    distance_interpolation = np.interp(angle_linspace, angles, distances, period=2 * np.pi)

    kernel_size_1 = 31
    if minimize:
        smoothed_distances = smooth_savgol_filter(
            distance_interpolation,
            kernel_size_1,
            minimize=True
        )
    elif maximize:
        smoothed_distances = smooth_savgol_filter(
            distance_interpolation,
            kernel_size_1,
            maximize=True
        )
    else:
        smoothed_distances = smooth_savgol_filter(
            distance_interpolation,
            kernel_size_1,
        )

    mask_radians = mask_angles_in_radian_range(angle_linspace)
    angles = np.array(angle_linspace)[~mask_radians]
    distances = np.array(smoothed_distances)[~mask_radians]
    return angles, distances


def get_normalized_from_coordinate(coordinate: Coordinate2d, img_width: int, img_height: int) -> Coordinate2dNormalized:
    return Coordinate2dNormalized(round(coordinate.x / img_width, 6), round(coordinate.y / img_height, 6))


def get_coordinate_from_normalized(coordinate_normalized: Coordinate2dNormalized, img_width: int, img_height: int) -> Coordinate2d:
    return Coordinate2d(np.uint32(coordinate_normalized.x * img_width), np.uint32(coordinate_normalized.y * img_height))


def get_distance_normalized(point1: Coordinate2dNormalized, point2: Coordinate2dNormalized) -> float:

    return np.linalg.norm(np.array((point1.x, point1.y)) - np.array((point2.x, point2.y)))


