import math
from typing import List, Union

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


def get_reverse_indices(list_to_index) -> List[int]:
    reverse_indices = list(range(len(list_to_index)))
    reverse_indices.reverse()
    return reverse_indices


def get_polynomial(x: List[Union[int, float]], y: List[Union[int, float]], degree=2):
    return Polynomial.fit(x, y, degree)


def flatten(list_of_lists: List[List]) -> list:
    flat_list = []
    for row in list_of_lists:
        flat_list += row
    return flat_list


def flatten_np_array_lists(list_of_np_arrays: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(list_of_np_arrays)


def split_list(input_list, chunk_size):
    """
    Takes an input list and the chunk size (in this case 5) and returns a list of lists where each sublist
    contains 5 items from the original list
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def get_extended_angles(angles: list):
    # assumes angles are in radians
    # assumes values are in [0, 2*pi]
    # add 10% tail of angles array to head and 10% head of angles to tail
    # is a necessary function for radian array extension
    # values will extend [0, 2*pi] after array extension
    return [
        angle - 2 * math.pi for angle in angles[int(-len(angles)*0.1):]
    ] + angles + [
        angle + 2 * math.pi for angle in angles[:int(len(angles)*0.1)]
    ]

# get_extended_angles and get_extended_list must return same list length with same input length


def get_extended_list(my_list: list):
    # add 10% tail of array to head and 10% head of array to tail
    return my_list[int(-len(my_list)*0.1):] + my_list + my_list[:int(len(my_list)*0.1)]


def smooth_savgol_filter(data: list, kernel_size_first: int, kernel_size_second: int = -1,
                         maximize: bool = False, minimize: bool = False):
    smoothed_data = savgol_filter(data, window_length=kernel_size_first, polyorder=1)
    if maximize:
        smoothed_data = np.maximum(smoothed_data, data)
    elif minimize:
        smoothed_data = np.minimum(smoothed_data, data)
    if kernel_size_second != -1:
        smoothed_data = savgol_filter(smoothed_data, window_length=kernel_size_second, polyorder=1)

    return smoothed_data


def get_cubic_interpolation(x, y):
    return CubicSpline(x, y)


def mask_first_occurrence_with_unique(a):
    mask = np.ones(len(a), dtype=bool)
    mask[np.unique(a, return_index=True)[1]] = False
    return mask


def mask_angles_in_radian_range(angles):
    return np.array([False if 0 <= angle <= 2 * math.pi else True for angle in angles])
