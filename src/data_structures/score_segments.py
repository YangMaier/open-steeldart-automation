from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np


class SegmentType(Enum):
    BULLS_EYE = 0
    BULL = 1
    INNER = 2
    TRIPLE = 3
    OUTER = 4
    DOUBLE = 5

    def equals(self, ft: 'SegmentType'):
        return self.value == ft.value


@dataclass
class ScoreSegment:
    contour: np.ndarray
    segment_type: SegmentType
    number: Union[None, int] = None

    def get_score(self):
        if self.segment_type == SegmentType.BULLS_EYE:
            return 50
        elif self.segment_type == SegmentType.BULL:
            return 25
        elif self.segment_type == SegmentType.TRIPLE:
            return self.number * 3
        elif self.segment_type == SegmentType.DOUBLE:
            return self.number * 2
        else:
            return self.number
