from typing import List

import numpy as np
from scipy.signal import savgol_filter

from data_structures.coordinates import Coordinate2d
from data_structures.coordinates_extended import CoordinateAngleDistance
from operations_on.coordinates_and_angles import get_endpoint_range


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


def get_interpolated_coordinates_with_smoothed_angles(angles, distances, base_coordinate: Coordinate2d, coordinate_quantity: int, savgol_window_length: int, mirror=True) -> List[CoordinateAngleDistance]:
    angles_shifted, angle_shift = get_angles_shifted_to_safe_range(angles)
    distance_linspace = np.linspace(distances[0], distances[-1], coordinate_quantity)
    angles_interpolated = np.interp(distance_linspace, distances, angles_shifted)
    if mirror:
        smoothed_angles = savgol_filter(angles_interpolated, savgol_window_length, 1, mode='mirror')
    else:
        smoothed_angles = savgol_filter(angles_interpolated, savgol_window_length, 1)
    smoothed_angles_shifted_back = shift_angles_back_to_original_range(smoothed_angles, angle_shift)
    new_coordinates: List[CoordinateAngleDistance] = get_endpoint_range(base_coordinate, smoothed_angles_shifted_back, distance_linspace)

    return new_coordinates
