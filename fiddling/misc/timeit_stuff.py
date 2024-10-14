from dataclasses import dataclass
import numpy as np


@dataclass
class Point:
    x: int
    y: int

    def get_distance_to_other(self, point: 'Point'):
        return np.linalg.norm(np.array((self.x, self.y)) - np.array((point.x, point.y)))

    def get_midpoint_to_other(self, point: 'Point'):
        return int((self.x + point.x) * 0.5), int((self.y + point.y) * 0.5)

    def get_line_extension_point_to_other(self, point: 'Point', length: float or int):
        # visualize a line from point a to b, then extend the line by times length, and return the end of that line
        end_x = int(self.x + (point.x - self.x) * length)
        end_y = int(self.y + (point.y - self.y) * length)
        return Point(end_x, end_y)


def get_distance(point1: Point, point2: Point) -> float:
    return np.linalg.norm(np.array((point1.x, point1.y)) - np.array((point2.x, point2.y)))


def is_point_in_tolerance(point1: Point, point2: Point) -> bool:
    return get_distance(point1, point2) < 3
