# BASIC TRANSITIONS

# F1 TRANSITIONS
# f1_inner_triple_transition_mid = get_transition_via_shapes(f1_triple, f1_inner, inner_triple_lb)
# f1_triple_outer_transition_mid = get_transition_via_shapes(f1_triple, f1_outer, triple_outer_lb)
# f1_outer_double_transition_mid = get_transition_via_shapes(f1_double, f1_outer, outer_double_lb)
# f1_double_board_transition_mid = get_shape_rim(f1_double, f1_outer.cp, length_base=double_board_lb)

# F2 TRANSITIONS
# just for calculation, do not add to any fields, will be added next loop
# f2_inner_triple_transition_mid = get_transition_via_shapes(f2_triple, f2_inner, inner_triple_lb)
# f2_triple_outer_transition_mid = get_transition_via_shapes(f2_triple, f2_outer, triple_outer_lb)
f2_outer_double_transition_mid = get_transition_via_shapes(f2_double, f2_outer, outer_double_lb)
f2_double_board_transition_mid = get_shape_rim(f2_double, f2_outer.cp, length_base=double_board_lb)

# F1 F2 TRANSITIONS
f1_f2_inner_midpoint = get_transition_via_shapes(f1_inner, f2_inner)
bulls_eye_bull_f1_f2_inner_midpoint_transition = get_line_extension_end_point(bulls_eye_cp, f1_f2_inner_midpoint, 0.12)
bulls_eye_cp_f1_f2_inner_midpoint_transition = get_line_extension_end_point(bulls_eye_cp, f1_f2_inner_midpoint, 0.25)
f1_f2_outer_midpoint = get_transition_via_shapes(f1_outer, f2_outer)

# INNER - TRIPLE TRANSITIONS PRE
# get the corner of four fields by using the midrange distance and angle from f1_f2_triple_midpoint
# because the midpoint angle is more precise
f1_f2_inner_triple_intersection = line_line_intersection(
    f1_f2_inner_midpoint, f1_f2_outer_midpoint,
    f1_inner_triple_transition_mid, f2_inner_triple_transition_mid
)
# f1_f2_triple_midpoint = get_transition_via_shapes(f1_triple, f2_triple)  # just for calculation, do not add to any field
f1_f2_triple_midpoint_angle = get_angle(f1_f2_inner_triple_intersection, bulls_eye_cp)
f1_f2_triple_midpoint_distance = get_distance(f1_f2_inner_triple_intersection, bulls_eye_cp)

# INNER - TRIPLE TRANSITIONS
f1_inner_triple_transition_angle = get_angle(f1_inner_triple_transition_mid, bulls_eye_cp)
f2_inner_triple_transition_angle = get_angle(f2_inner_triple_transition_mid, bulls_eye_cp)

board_cp_f1_inner_triple_distance = get_distance(bulls_eye_cp, f1_inner_triple_transition_mid)
board_cp_f2_inner_triple_distance = get_distance(bulls_eye_cp, f2_inner_triple_transition_mid)

# [0] and [4] are already known
# angle_range = get_angle_range(f1_inner_triple_transition_mid_angle, f2_inner_triple_transition_mid_angle)
# distance_range = get_distance_range(board_cp_f1_distance_mid, board_cp_f2_distance_mid)

# distance range needs some influence of f1_f2_triple_transition distance
# distance_diff = abs(f1_f2_triple_midpoint_distance - distance_range[2])
f1_and_f2_calc_distance_single_transition_mod = 1.01
f1_f2_calc_distance_transition_mod = 1.015

f1_inner_triple_transition_calc_distance = board_cp_f1_inner_triple_distance * f1_and_f2_calc_distance_single_transition_mod
f2_inner_triple_transition_calc_distance = board_cp_f2_inner_triple_distance * f1_and_f2_calc_distance_single_transition_mod
f1_f2_inner_triple_transition_calc_distance = f1_f2_triple_midpoint_distance * f1_f2_calc_distance_transition_mod
# get another point for f1 and f2 inner and triple calc shape
f1_inner_triple_transition_calc = get_endpoint_via(bulls_eye_cp, f1_inner_triple_transition_angle,
                                                   f1_inner_triple_transition_calc_distance)
# add to 2 fields
f1_inner.score_border.add_point(f1_inner_triple_transition_calc)
f1_triple.score_border.add_point(f1_inner_triple_transition_calc)
f2_inner_triple_transition_calc = get_endpoint_via(bulls_eye_cp, f2_inner_triple_transition_angle,
                                                   f2_inner_triple_transition_calc_distance)
# add to 2 fields
f2_inner.score_border.add_point(f2_inner_triple_transition_calc)
f2_triple.score_border.add_point(f2_inner_triple_transition_calc)
f1_f2_inner_triple_transition_calc = get_endpoint_via(bulls_eye_cp, f1_f2_triple_midpoint_angle,
                                                      f1_f2_inner_triple_transition_calc_distance)
# add to 4 fields
f1_inner.score_border.add_point(f1_f2_inner_triple_transition_calc)
f1_triple.score_border.add_point(f1_f2_inner_triple_transition_calc)
f2_inner.score_border.add_point(f1_f2_inner_triple_transition_calc)
f2_triple.score_border.add_point(f1_f2_inner_triple_transition_calc)

# TRIPLE - OUTER TRANSITIONS
f1_transition_mid_angle = get_angle(f1_triple_outer_transition_mid, bulls_eye_cp)
f2_transition_mid_angle = get_angle(f2_triple_outer_transition_mid, bulls_eye_cp)
board_cp_f1_distance_mid = get_distance(bulls_eye_cp, f1_triple_outer_transition_mid)
board_cp_f2_distance_mid = get_distance(bulls_eye_cp, f2_triple_outer_transition_mid)

# [0] and [4] are already known
angle_range = get_angle_range(f1_transition_mid_angle, f2_transition_mid_angle)
distance_range = get_distance_range(board_cp_f1_distance_mid, board_cp_f2_distance_mid)

# distance range needs some influence of f1_f2_triple_transition distance
distance_diff = abs(f1_f2_triple_midpoint_distance - distance_range[2])
f1_and_f2_calc_distance_single_transition_mod = 0.01 * distance_diff
f1_f2_calc_distance_transition_mod = 0.012 * distance_diff

f1_triple_outer_transition_calc_distance = distance_range[1] + f1_and_f2_calc_distance_single_transition_mod
f2_triple_outer_transition_calc_distance = distance_range[3] + f1_and_f2_calc_distance_single_transition_mod
f1_f2_triple_outer_transition_calc_distance = distance_range[2] + f1_f2_calc_distance_transition_mod
# get another point for f1 and f2 inner and triple calc shape
f1_triple_outer_transition_calc = get_endpoint_via(bulls_eye_cp, angle_range[1],
                                                   f1_triple_outer_transition_calc_distance)
# add to 2 fields
f1_triple.score_border.add_point(f1_triple_outer_transition_calc)
f1_outer.score_border.add_point(f1_triple_outer_transition_calc)
f2_triple_outer_transition_calc = get_endpoint_via(bulls_eye_cp, angle_range[3],
                                                   f2_triple_outer_transition_calc_distance)
# add to 2 fields
f2_triple.score_border.add_point(f2_triple_outer_transition_calc)
f2_outer.score_border.add_point(f2_triple_outer_transition_calc)
f1_f2_triple_outer_transition_calc = get_endpoint_via(bulls_eye_cp, f1_f2_triple_midpoint_angle,
                                                      f1_f2_triple_outer_transition_calc_distance)
# add to 4 fields
f1_triple.score_border.add_point(f1_f2_triple_outer_transition_calc)
f1_outer.score_border.add_point(f1_f2_triple_outer_transition_calc)
f2_triple.score_border.add_point(f1_f2_triple_outer_transition_calc)
f2_outer.score_border.add_point(f1_f2_triple_outer_transition_calc)

# OUTER - DOUBLE TRANSITION PRE
# get the corner of four fields by using the midrange distance and angle from f1_f2_triple_midpoint
# because the midpoint angle is more precise
f1_f2_double_midpoint = get_transition_via_shapes(f1_double, f2_double)  # just for calculation, do not add to any field
f1_f2_double_midpoint_angle = get_angle(f1_f2_double_midpoint, bulls_eye_cp)
f1_f2_double_midpoint_distance = get_distance(f1_f2_double_midpoint, bulls_eye_cp)

# OUTER - DOUBLE TRANSITION
f1_transition_mid_angle = get_angle(f1_outer_double_transition_mid, bulls_eye_cp)
f2_transition_mid_angle = get_angle(f2_outer_double_transition_mid, bulls_eye_cp)
board_cp_f1_distance_mid = get_distance(bulls_eye_cp, f1_outer_double_transition_mid)
board_cp_f2_distance_mid = get_distance(bulls_eye_cp, f2_outer_double_transition_mid)

# [0] and [4] are already known
angle_range = get_angle_range(f1_transition_mid_angle, f2_transition_mid_angle)
distance_range = get_distance_range(board_cp_f1_distance_mid, board_cp_f2_distance_mid)

# distance range needs some influence of f1_f2_double_transition distance
distance_diff = abs(f1_f2_double_midpoint_distance - distance_range[2])
f1_and_f2_calc_distance_single_transition_mod = 0.06 * distance_diff
f1_f2_calc_distance_transition_mod = 0.08 * distance_diff

f1_outer_double_transition_calc_distance = distance_range[1] + f1_and_f2_calc_distance_single_transition_mod
f2_outer_double_transition_calc_distance = distance_range[3] + f1_and_f2_calc_distance_single_transition_mod
f1_f2_outer_double_transition_calc_distance = distance_range[2] + f1_f2_calc_distance_transition_mod
# get another point for f1 and f2 inner and triple calc shape
f1_outer_double_transition_calc = get_endpoint_via(bulls_eye_cp, angle_range[1],
                                                   f1_outer_double_transition_calc_distance)
# add to 2 fields
f1_outer.score_border.add_point(f1_outer_double_transition_calc)
f1_double.score_border.add_point(f1_outer_double_transition_calc)
f2_outer_double_transition_calc = get_endpoint_via(bulls_eye_cp, angle_range[3],
                                                   f2_outer_double_transition_calc_distance)
# add to 2 fields
f2_outer.score_border.add_point(f2_outer_double_transition_calc)
f2_double.score_border.add_point(f2_outer_double_transition_calc)
f1_f2_outer_double_transition_calc = get_endpoint_via(bulls_eye_cp, f1_f2_double_midpoint_angle,
                                                      f1_f2_outer_double_transition_calc_distance)
# add to 4 fields
f1_outer.score_border.add_point(f1_f2_outer_double_transition_calc)
f1_double.score_border.add_point(f1_f2_outer_double_transition_calc)
f2_outer.score_border.add_point(f1_f2_outer_double_transition_calc)
f2_double.score_border.add_point(f1_f2_outer_double_transition_calc)

# DOUBLE - BOARD TRANSITION
f1_transition_mid_angle = get_angle(f1_double_board_transition_mid, bulls_eye_cp)
f2_transition_mid_angle = get_angle(f2_double_board_transition_mid, bulls_eye_cp)
board_cp_f1_distance_mid = get_distance(bulls_eye_cp, f1_double_board_transition_mid)
board_cp_f2_distance_mid = get_distance(bulls_eye_cp, f2_double_board_transition_mid)

# [0] and [4] are already known
angle_range = get_angle_range(f1_transition_mid_angle, f2_transition_mid_angle)
distance_range = get_distance_range(board_cp_f1_distance_mid, board_cp_f2_distance_mid)

# distance range needs some influence of f1_f2_double_transition distance
distance_diff = abs(f1_f2_double_midpoint_distance - distance_range[2])
f1_and_f2_calc_distance_single_transition_mod = 0.01 * distance_diff
f1_f2_calc_distance_transition_mod = 0.012 * distance_diff

f1_double_board_transition_calc_distance = distance_range[1] + f1_and_f2_calc_distance_single_transition_mod
f2_double_board_transition_calc_distance = distance_range[3] + f1_and_f2_calc_distance_single_transition_mod
f1_f2_double_board_transition_calc_distance = distance_range[2] + f1_f2_calc_distance_transition_mod
# get another point for f1 and f2 inner and double calc shape
f1_double_board_transition_calc = get_endpoint_via(bulls_eye_cp, angle_range[1],
                                                   f1_double_board_transition_calc_distance)
# add to 1 field
f1_double.score_border.add_point(f1_double_board_transition_calc)
f2_double_board_transition_calc = get_endpoint_via(bulls_eye_cp, angle_range[3],
                                                   f2_double_board_transition_calc_distance)
# add to 1 field
f2_double.score_border.add_point(f2_double_board_transition_calc)
f1_f2_double_board_transition_calc = get_endpoint_via(bulls_eye_cp, f1_f2_double_midpoint_angle,
                                                      f1_f2_double_board_transition_calc_distance)
# add to 2 fields
f1_double.score_border.add_point(f1_f2_double_board_transition_calc)
f2_double.score_border.add_point(f1_f2_double_board_transition_calc)

# SEGMENT CALC POINT SAFES

# F1 TRANSITIONS
f1_inner.score_border.add_point(f1_inner_triple_transition_mid)
f1_triple.score_border.add_point(f1_inner_triple_transition_mid)

f1_triple.score_border.add_point(f1_triple_outer_transition_mid)
f1_outer.score_border.add_point(f1_triple_outer_transition_mid)

f1_outer.score_border.add_point(f1_outer_double_transition_mid)
f1_double.score_border.add_point(f1_outer_double_transition_mid)

f1_double.score_border.add_point(f1_double_board_transition_mid)

# F1 F2 INNER TRANSITION
f1_inner.score_border.add_point(f1_f2_inner_midpoint)
f2_inner.score_border.add_point(f1_f2_inner_midpoint)

# F1 F2 BULLS EYE - BULL TRANSITION
dart_board.bulls_eye_segment.add_calculated_border_point(bulls_eye_bull_f1_f2_inner_midpoint_transition)
dart_board.bull_segment.add_calculated_border_point(bulls_eye_bull_f1_f2_inner_midpoint_transition)

# F1 F2 BULL INNER TRANSITION
f1_inner.score_border.add_point(bulls_eye_cp_f1_f2_inner_midpoint_transition)
f2_inner.score_border.add_point(bulls_eye_cp_f1_f2_inner_midpoint_transition)
dart_board.bull_segment.score_border.add_point(bulls_eye_cp_f1_f2_inner_midpoint_transition)

# F1 F2 OUTER TRANSITION
f1_outer.score_border.add_point(f1_f2_outer_midpoint)
f2_outer.score_border.add_point(f1_f2_outer_midpoint)