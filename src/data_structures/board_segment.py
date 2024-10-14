from dataclasses import dataclass, field
from typing import List

import numpy as np

from data_structures.coordinates_extended import CoordinateAngleDistance
from data_structures.score_segments import SegmentType


@dataclass
class BoardSegment:
    center_cad: CoordinateAngleDistance
    contour: np.ndarray
    contour_cads: List[CoordinateAngleDistance]
    low_angle_side_cad: CoordinateAngleDistance
    high_angle_side_cad: CoordinateAngleDistance
    segment_type: SegmentType
    number = None  # will be added later if it's a double segment
