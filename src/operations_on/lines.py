import math

import cv2 as cv
import numpy as np

from data_structures.coordinates import Coordinate2d
from operations_on.coordinates import get_distance


# def draw_line(img, cols, lefty, righty, color, thickness=2):
#     cv.line(img, (cols - 1, righty), (0, lefty), color, thickness)


def draw_line(img, point1: Coordinate2d, point2: Coordinate2d, color=(255, 255, 255), thickness=2):
    cv.line(img, (point1.x, point1.y), (point2.x, point2.y), color, thickness)


def calculate_line_min_distance(a, b, c):
    """
    Calculates the minimum distance from point c to the line defined by points a and b.

    Args:
        a (tuple): The coordinates of point a.
        b (tuple): The coordinates of point b.
        c (tuple): The coordinates of point c.

    Returns:
        float: The minimum distance from point c to the line defined by points a and b.

    Description:
        This function calculates the minimum distance from point `c` to the line defined by points `a` and `b`. It uses the formula for the distance from a point to a line, which is the length of the perpendicular line from `c` to the line defined by `a` and `b`, divided by the length of the line segment defined by `a` and `b`.

    Example:
        calculate_min_distance((0, 0), (3, 4), (1, 2))
        # Returns: 1.0
    """
    # Calculate the slope and y-intercept of the line defined by points a and b
    slope = (b[1] - a[1]) / (b[0] - a[0])
    y_intercept = a[1] - slope * a[0]

    # Calculate the distance from point c to the line
    distance = abs((c[1] - slope * c[0] - y_intercept) / math.sqrt(1 + slope**2))

    return distance


def calculate_min_distance_from_line(a, b, c):
    p1 = np.array(a)
    p2 = np.array(b)
    p3 = np.array(c)
    d = np.abs(np.linalg.norm(np.cross(p2 - p1, p1 - p3))) / get_distance(a, b)
    return d
