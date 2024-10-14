from dataclasses import dataclass, field
from typing import List

import numpy as np

from data_structures.coordinates import Coordinate2d
from operations_on.angles import get_angle_diff, get_mid_angle


@dataclass
class EdgeValues:
    edge_points: List[Coordinate2d]
    c_min_angle: float
    c_max_angle: float
    mid_angle: float = field(init=False)
    c_min_angle_new: float = field(init=False)
    c_max_angle_new: float = field(init=False)

    def __post_init__(self):
        self.mid_angle = self.get_mid_angle()

    def get_mid_angle(self):
        return get_mid_angle(self.c_min_angle, self.c_max_angle)

    def get_angle_diff(self):
        return get_angle_diff(self.c_min_angle, self.c_max_angle)


@dataclass
class AngleDistanceInterpolation:
    angles: np.ndarray
    distances: np.ndarray
