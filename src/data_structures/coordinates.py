
from dataclasses import dataclass

import numpy as np


@dataclass
class Coordinate2d:
    """Classic 2d point in an image"""
    # A precise datatype is always faster than python integer
    # I would like to put the datatype to uint16,
    # but some opencv functions don't like uint16 and some don't like uint32.
    # That's why it is int32
    x: np.int32
    y: np.int32

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def as_np_arr(self):
        return np.array([self.x, self.y])


@dataclass
class Coordinate2dNormalized:
    """Representing a relative position in an image
    x = y = 0 is the top left corner of the image
    x = y = 1 is the bottom right corner of the image
    """
    x: np.float32  # pixel_x / img_width
    y: np.float32  # pixel_y / img_height

    def __post_init__(self):
        self.x = round(self.x, 6)
        self.y = round(self.y, 6)

    def __repr__(self):
        return f"({self.x}, {self.y})"

