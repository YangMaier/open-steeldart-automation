import numpy as np

from data_structures.board_segment_old import BoardSegment
from data_structures.coordinates import Coordinate2d
from operations_on.coordinates import get_mid_coordinate


def get_midpoint_to_other(board_field_1: BoardSegment, board_field_2: BoardSegment) -> Coordinate2d:
    return Coordinate2d(
        int((board_field_1.cp.x + board_field_2.cp.x) * 0.5),
        int((board_field_1.cp.y + board_field_2.cp.y) * 0.5)
    )


def get_transition_via_shapes(bf_1: BoardSegment, bf_2: BoardSegment) -> Coordinate2d:
    bf1_shape_rim: Coordinate2d = bf_1.get_shape_edge_towards(bf_2.cp)
    bf2_shape_rim: Coordinate2d = bf_2.get_shape_edge_towards(bf_1.cp)
    return get_mid_coordinate(bf1_shape_rim, bf2_shape_rim)

#
# def get_transition_via_shapes(bf_1: BoardField, bf_2: BoardField, bigger_loses=False) -> Point2d:
#     # iterative approach
#
#     search_iterations = 6
#     p1: Point2d
#     p2: Point2d
#     if bigger_loses:
#         # return point where the bigger shape ends
#         bigger: BoardField
#         smaller: BoardField
#         if bf_1.contour.size > bf_2.contour.size:
#             bigger = bf_1
#             smaller = bf_2
#         else:
#             bigger = bf_2
#             smaller = bf_1
#         p1 = bigger.cp
#         p2 = smaller.cp
#         for _ in range(search_iterations):
#             p_range = np.linspace((p1.x, p1.y), (p2.x, p2.y), 4, dtype=int)
#             p_range = [Point2d(int(x), int(y)) for x, y in p_range]
#             distances = [
#                 (bf_1.get_contour_distance_to(p_range[0]), p_range[0]),
#                 (bf_1.get_contour_distance_to(p_range[1]), p_range[1]),
#                 (bf_1.get_contour_distance_to(p_range[2]), p_range[2]),
#                 (bf_1.get_contour_distance_to(p_range[3]), p_range[3])
#             ]
#             distances_positive = [dist for dist in distances if dist[0] >= 0]
#             distances_negative = [dist for dist in distances if dist[0] < 0]
#             distances_positive = sorted(distances_positive, key=lambda x: x[0])
#             distances_negative = sorted(distances_negative, key=lambda x: x[0], reverse=True)
#
#             p1 = distances_negative[0][1]
#             p2 = distances_positive[0][1]
#
#             if p1 == p2:
#                 break
#
#     else:
#         p1 = bf_1.cp
#         p2 = bf_2.cp
#         for _ in range(search_iterations):
#             p_range = np.linspace((p1.x, p1.y), (p2.x, p2.y), 4, dtype=int)
#             p_range = [Point2d(int(x), int(y)) for x, y in p_range]
#             distances = [
#                 (abs(bf_1.get_contour_distance_to(p_range[0]) - bf_2.get_contour_distance_to(p_range[0])), p_range[0]),
#                 (abs(bf_1.get_contour_distance_to(p_range[1]) - bf_2.get_contour_distance_to(p_range[1])), p_range[1]),
#                 (abs(bf_1.get_contour_distance_to(p_range[2]) - bf_2.get_contour_distance_to(p_range[2])), p_range[2]),
#                 (abs(bf_1.get_contour_distance_to(p_range[3]) - bf_2.get_contour_distance_to(p_range[3])), p_range[3])
#             ]
#             d_sorted = sorted(distances, key=lambda x: x[0])
#             p1 = d_sorted[0][1]
#             p2 = d_sorted[1][1]
#
#             if abs(p1.x - p2.x) + abs(p1.y - p2.y) < 4:
#                 break
#
#     return get_midpoint(p1, p2)

    # midpoint = get_midpoint(board_field_1.cp, board_field_2.cp)
    # midpoint_range = get_range_in_between_points(board_field_1.cp, midpoint, length_base=length_base)
    # paired_distances = []
    # for i, p in enumerate(midpoint_range):
    #     dist_1 = board_field_1.get_contour_distance_to(p)
    #     dist_2 = board_field_2.get_contour_distance_to(p)
    #     # Best possible outcome is when the two distances are equal.
    #     # That's when the point is exactly between the shapes.
    #     # In that case, the distance difference is 0.
    #     dist_diff = abs(dist_1 - dist_2)
    #     paired_distances.append((p, dist_diff))

    # return sorted(paired_distances, key=lambda x: x[1])[0][0]  # return point with minimum distance sum to shapes
