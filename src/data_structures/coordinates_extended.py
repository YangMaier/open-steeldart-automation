from dataclasses import dataclass, field
from typing import List, Union

import numpy as np

from data_structures.coordinates import Coordinate2d
from data_structures.score_segments import SegmentType


@dataclass
class CoordinateContour:
    coordinate: Coordinate2d
    contour: np.ndarray


@dataclass
class CoordinateAngleDistance:
    coordinate: Coordinate2d
    angle: np.float32
    distance: np.float32

    def __hash__(self):
        return hash(self.angle)


@dataclass
class CoordinateAngleDistanceContour(CoordinateAngleDistance):
    # coordinate angle and distance are used as center coordinate and that coordinates angle and distance from board center
    contour: np.ndarray


@dataclass
class CoordinateAngleDistanceContourNumber(CoordinateAngleDistanceContour):
    # coordinate angle and distance are used as center coordinate and that coordinates angle and distance from board center
    # the number corresponds to the number of the radial section
    number: np.uint8


@dataclass
class RadialSectionRingIntersections:
    # "clockwise" corners always starting with the nearest left corner from the bulls-eye's viewpoint
    cads: List[CoordinateAngleDistance]  # should have len 4
    number: np.uint8

    def get_by_distance(self) -> List[CoordinateAngleDistance]:
        self.cads.sort(key=lambda p: p.distance)
        return self.cads


@dataclass
class RadialSectionRingIntersectionsEllipse:
    # "clockwise" corners always starting with the nearest left corner from the bulls-eye's viewpoint
    double_corner: CoordinateAngleDistance
    ellipse_corner: CoordinateAngleDistance
    cads_original: List[CoordinateAngleDistance]
    coordinates_calculated: Union[List[Coordinate2d], None] = None  # should have len 5, is filled later

    def get_by_distance_as_cads(self) -> List[CoordinateAngleDistance]:
        self.cads_original.sort(key=lambda p: p.distance)

        return self.cads_original

    def get_by_distance_as_np(self) -> np.ndarray:
        coordinates_np = np.asarray([(cad.coordinate.x, cad.coordinate.y) for cad in self.get_by_distance_as_cads()])

        return coordinates_np

    def get_calculated_as_np(self):
        coordinates_np = np.asarray([(c.x, c.y) for c in self.coordinates_calculated])

        return coordinates_np


@dataclass
class CoordinateGroup:
    cads: List[CoordinateAngleDistance] = field(default_factory=list)
    mean_angle: np.float32 = None
    mean_distance: np.float32 = None

    def __post_init__(self):
        self.mean_angle = np.mean([p.angle for p in self.cads])
        self.mean_distance = np.mean([p.distance for p in self.cads])


@dataclass
class CoordinateSegmentTypeNumber:
    coordinate: Coordinate2d
    segment_type: SegmentType
    number: np.uint8 or None
