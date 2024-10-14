import abc
from abc import ABC
from dataclasses import dataclass, field
from typing import List

import numpy as np

from data_structures.coordinates import Coordinate2d
from data_structures.coordinates_extended import CoordinateAngleDistance
from fiddling.misc.sort_points_clockwise import get_clockwise


@dataclass
class ScoreSegmentOld(ABC):
    # field shape used for score calculation
    # extracted/calculated from board center, contour and other contours

    @abc.abstractmethod
    def get_points(self) -> List[Coordinate2d]:
        raise NotImplementedError

    def get_points_as_np(self) -> np.ndarray:
        return np.array([p.as_np_arr() for p in self.get_points()])


@dataclass
class BullOrBullseyeScoreSegment(ScoreSegmentOld):
    _border_points: [Coordinate2d] = field(default_factory=list)

    def get_points(self):
        return np.array([[p.x, p.y] for p in self.get_points_clockwise()])

    def get_points_clockwise(self):
        # sort border points clockwise, so they can be used as an opencv.Contour
        border_points = [(p.x, p.y) for p in self._border_points]
        self._border_points = get_clockwise(border_points)
        self._border_points = [Coordinate2d(x, y) for x, y in self._border_points]
        return self._border_points

    def add_point(self, point: Coordinate2d):
        self._border_points.append(point)

    def add_points(self, points: [Coordinate2d]):
        self._border_points.extend(points)


@dataclass
class NumberedScoreSegment(ScoreSegmentOld):
    # points are numbered clockwise.
    # c_1 inner edge min angle to c2 outer edge min angle,
    # c_3 outer edge max angle to c4 inner edge max angle
    # 5 points are used for the curved borders
    # for simplicity’s sake also the inner border of the inner field has 5 points even tho that's too many points
    c_1: Coordinate2d = field(init=False, default=None)  # inner edge min angle
    c_1_1_4: Coordinate2d = field(init=False, default=None)
    c_1_4: Coordinate2d = field(init=False, default=None)
    c_1_4_4: Coordinate2d = field(init=False, default=None)
    c_4: Coordinate2d = field(init=False, default=None)  # inner edge max angle

    c_2: CoordinateAngleDistance = field(init=False, default=None)  # outer edge min angle
    c_2_2_3: CoordinateAngleDistance = field(init=False, default=None)
    c_2_3: CoordinateAngleDistance = field(init=False, default=None)
    c_2_3_3: CoordinateAngleDistance = field(init=False, default=None)
    c_3: CoordinateAngleDistance = field(init=False, default=None)  # outer edge max angle

    def get_points(self):
        return np.array([self.c_1, self.c_2, self.c_2_2_3, self.c_2_3, self.c_2_3_3, self.c_3, self.c_4, self.c_1_4_4, self.c_1_4, self.c_1_1_4])
