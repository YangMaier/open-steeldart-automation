from collections import deque
import numpy as np


class SegmentCenterPointAccumulator:
    def __init__(self, center_point):
        self._center_points = deque(maxlen=20)
        self.add_frame_point(center_point)
        self._is_board_center = False
        self._last_known_mcp = 0
        self._point_was_added_this_frame = False

    def set_board_center(self):
        self._is_board_center = True

    def get_board_center(self):
        return self._is_board_center

    def add_frame_point(self, point):
        self._center_points.append(point)
        self._last_known_mcp = self.get_mean_cp()
        self.set_point_was_added_this_frame()#

    def add_empty_frame_point(self):
        self._center_points.append((np.NAN, np.NAN))  # works with np.nanmean

    def get_mean_cp(self):
        mean = np.nanmean(self._center_points, axis=0)
        return int(mean[0]), int(mean[1])

    def translate_to_cpad(self, offset_x, offset_y, offset_y_multiplier):
        mcp = np.mean(self._center_points, axis=0)

        # apply offset first, then multiplier
        return int(mcp[0] + offset_x), (int(mcp[1] + offset_y) * offset_y_multiplier)

    def get_distance_to(self, point):
        return np.linalg.norm(np.array(self.get_mean_cp()) - np.array(point))

    def in_neighborhood_of(self, point, radius=5):
        return self.get_distance_to(point) < radius

    def represents_consistent_contour(self):
        # the contour is consistent if:
        # - the cp deque is full
        # - a maximum of two empty points are currently stored
        queue_full = len(self._center_points) == self._center_points.maxlen
        max_two_empty_points = sum(np.ma.masked_invalid(self._center_points).mask[:, 0]) < 3
        return queue_full and max_two_empty_points

    def set_point_was_added_this_frame(self):
        self._point_was_added_this_frame = True

    def get_point_was_added_this_frame(self):
        return self._point_was_added_this_frame

    def reset_point_was_added_this_frame(self):
        self._point_was_added_this_frame = False

    def only_nan_points_left(self):
        return sum(np.ma.masked_invalid(self._center_points).mask[:, 0]) == self._center_points.maxlen
