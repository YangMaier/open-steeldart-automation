import math

import numpy as np


def get_mid_angle(angle1, angle2):
    # assumes the angle to get between is in radians and angle1 < angle2
    # if angle1 is not smaller than angle2, 2*pi is between the two angles

    if angle1 > 5 and angle2 < 1:
        angle2 += 2 * math.pi
    elif angle2 > 5 and angle1 < 1:
        angle1 += 2 * math.pi

    angle_diff = max(angle1, angle2) - min(angle1, angle2)
    angle_between = min(angle1, angle2) + angle_diff / 2

    return angle_between % (2 * math.pi)

    #
    # angle1 = 2 * math.pi if angle1 == 0 else angle1
    # angle2 = 2 * math.pi if angle2 == 0 else angle2
    #
    # if angle1 > angle2 and angle1 - angle2 > 1:
    #     # shift angles to safe range and back
    #     angle1_to_rad_max = 2 * math.pi - angle1
    #     angle1 = 0
    #     angle2 += angle1_to_rad_max
    #     angle_diff = angle2 - angle1
    #     angle_between = angle1 + angle_diff / 2
    #     angle_between -= angle1_to_rad_max
    #     angle_between %= 2 * math.pi
    # else:
    #     angle_diff = abs(angle2 - angle1)
    #     angle_between = min(angle1, angle2) + angle_diff / 2
    #
    # return angle_between


def get_angle_diff(angle1, angle2):
    # in radians
    # possible cases:
    # angle1 = 5.1, angle2 = 5.5, normal case
    # angle1 = 6.2, angle2 = 0.1, edge case

    if angle1 > 5 and angle2 < 1:
        angle2 += 2 * math.pi
    elif angle2 > 5 and angle1 < 1:
        angle1 += 2 * math.pi

    angle_diff = max(angle1, angle2) - min(angle1, angle2)
    # angle1 = 2 * math.pi if angle1 == 0 else angle1
    # angle2 = 2 * math.pi if angle2 == 0 else angle2
    #
    # if angle1 > angle2:
    #     # shift angles to safe range and back
    #     angle1_to_rad_max = 2 * math.pi - angle1
    #     angle1 = 0
    #     angle2 += angle1_to_rad_max
    #     angle_diff = angle2 - angle1
    #     angle_between = angle1 + angle_diff / 2
    #     angle_between -= angle1_to_rad_max
    #     angle_between %= 2 * math.pi
    # else:
    #     angle_diff = angle2 - angle1

    return angle_diff


def get_angle_range(angle1, angle2, num=5):

    if angle1 > 5 and angle2 < 1:
        angle2 += 2 * math.pi
    elif angle2 > 5 and angle1 < 1:
        angle1 += 2 * math.pi
    #
    # if angle2 - angle1 > 1:
    #     angle_temp = angle1
    #     angle1 = angle2
    #     angle2 = angle_temp
    angle_range = None
    # in radians
    # assumes angle1 is clockwise before angle2
    # if angle1 > angle2:  # only the case if we want to build a linspace through 2pi
    #     # shift angles to safe range and back
    #     angle1_to_rad_max = 2 * math.pi - angle1
    #     angle1 = 0
    #     angle2 += angle1_to_rad_max
    #     angle_range = np.linspace(angle1, angle2, num=num)
    #     angle_range -= angle1_to_rad_max
    #     angle_range %= 2 * math.pi
    # else:
    angle_range = np.linspace(angle1, angle2, num=num)
    angle_range = angle_range % (2 * math.pi)

    return angle_range


def is_angle_in_range(angle, angle1, angle2):
    if angle1 > 5 and angle2 < 1:
        angle2 += 2 * math.pi
        if angle < 1:
            angle += 2 * math.pi
    elif angle2 > 5 and angle1 < 1:
        angle1 += 2 * math.pi
        if angle < 1:
            angle1 += 2 * math.pi

    return min(angle1, angle2) <= angle <= max(angle1, angle2)


def get_min_angle(angle_range):
    if min(angle_range) < 1 and max(angle_range) > 5:
        angle_range = [angle for angle in angle_range if angle > 4]

    return min(angle_range)


def get_max_angle(angle_range):
    if min(angle_range) < 1 and max(angle_range) > 5:
        angle_range = [angle for angle in angle_range if angle < 3]

    return max(angle_range)


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y
