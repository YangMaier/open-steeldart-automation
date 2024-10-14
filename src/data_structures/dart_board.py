import copy
from dataclasses import dataclass, field
from typing import List, Type

import cv2 as cv
import numpy as np

from data_structures.board_segment_old import BoardSegment, AssociatedBoardSegments, InnerSegment, \
    TripleSegment, OuterSegment, DoubleSegment, BullsEyeSegment, BullSegment, BullBullsEyeSegment
from data_structures.interpolation import AngleDistanceInterpolation
from data_structures.letters import LetterToDetermine
from data_structures.score_segments import SegmentType
from data_structures.coordinates import Coordinate2d
from operations_on.angles import get_mid_angle, get_angle_range
from operations_on.contours import get_contour_distance_to, get_interpolations
from operations_on.letters import get_number_contour, match_letter_contours
from operations_on.lists import get_reverse_indices
from operations_on.coordinates import get_line_extension_end_point
from operations_on.coordinates_and_angles import get_endpoint_range_with_distance_interpolation, \
    get_endpoint_via_interp
from personal_settings import DetectionSettings


@dataclass
class DartBoard:
    all_contours: List[BoardSegment] = field(default_factory=list)
    bulls_eye_segment: BullsEyeSegment = field(init=False)
    bull_segment: BullSegment = field(init=False)
    mean_isolated_board_center_point: Coordinate2d = field(init=False)
    inner_fields: List[BoardSegment] = field(default_factory=list)
    not_inner_fields: List[BoardSegment] = field(default_factory=list)
    unsorted_associated_fields: List[List[BoardSegment]] = field(default_factory=list)
    associated_number_segments: List[AssociatedBoardSegments] = field(default_factory=list)
    bulls_eye_interp: AngleDistanceInterpolation = field(init=False)
    bull_interp: AngleDistanceInterpolation = field(init=False)
    inner_interp_e_1_4: AngleDistanceInterpolation = field(init=False)
    inner_interp_e_2_3: AngleDistanceInterpolation = field(init=False)
    triple_interp_e_1_4: AngleDistanceInterpolation = field(init=False)
    triple_interp_e_2_3: AngleDistanceInterpolation = field(init=False)
    outer_interp_e_1_4: AngleDistanceInterpolation = field(init=False)
    outer_interp_e_2_3: AngleDistanceInterpolation = field(init=False)
    double_interp_e_1_4: AngleDistanceInterpolation = field(init=False)
    double_interp_e_2_3: AngleDistanceInterpolation = field(init=False)

    # detection settings
    cam_resolution = (None, None)
    isolated_distance_center_to_inner = None

    def add_settings(self, detection_settings: Type[DetectionSettings]):
        self.cam_resolution = (detection_settings.img_height, detection_settings.img_width)
        self.isolated_distance_center_to_inner = detection_settings.isolated_distance_center_to_inner


    def add_contours(self, new_contours: List[BoardSegment]):
        self.all_contours.extend(new_contours)

    def sorted_by_cp_x(self) -> List[BoardSegment]:
        return sorted(self.all_contours, key=lambda board_field: board_field.cp.x)

    def sorted_by_cp_y(self) -> List[BoardSegment]:
        return sorted(self.all_contours, key=lambda board_field: board_field.cp.y)

    def sorted_by_isolated_distance_from_board_center(self) -> List[BoardSegment]:
        return sorted(self.all_contours, key=lambda x: x.isolated_distance_from_board_center)

    def calculate_isolated_cps(self):
        # isolate the points from the real pixel values by subtracting the overall min x and y value from each point
        # stretch the point coordinates to approximately fit a square
        cps_sorted_x = self.sorted_by_cp_x()
        cp_x_min = cps_sorted_x[0].cp.x
        cp_x_max = cps_sorted_x[-1].cp.x
        cps_sorted_y = self.sorted_by_cp_y()
        cp_y_min = cps_sorted_y[0].cp.y
        cp_y_max = cps_sorted_y[-1].cp.y

        if cp_x_max > cp_y_max:  # should always the case but im not sure
            for cnt in self.all_contours:
                iso_cp = Coordinate2d(int(cnt.cp.x - cp_x_min), int((cnt.cp.y - cp_y_min) * (cp_x_max / cp_y_max)))
                cnt.set_isolated_cp(iso_cp)
        else:
            for cnt in self.all_contours:
                iso_cp = Coordinate2d(int((cnt.cp.x - cp_x_min) * (cp_y_max / cp_x_max)), int(cnt.cp.y - cp_y_min))
                cnt.set_isolated_cp(iso_cp)

    def calculate_mean_isolated_board_point(self):
        mean_board_center = np.mean([(cnt.isolated_cp.x, cnt.isolated_cp.y) for cnt in self.all_contours], axis=0)
        self.mean_isolated_board_center_point = Coordinate2d(int(mean_board_center[0]), int(mean_board_center[1]))

    def sort_inner_fields_clockwise(self):
        # sort the inner fields clockwise, starting point is min(y)
        inner_field_cps = sorted(self.inner_fields,
                                 key=lambda board_field: board_field.isolated_cp.y)  # sort by y value
        inner_field_cps_right = copy.deepcopy(inner_field_cps)
        inner_field_cps_left = copy.deepcopy(inner_field_cps)
        # overwrite current inner fields list and insert starting point
        self.inner_fields = [inner_field_cps[0]]
        # Because we only look at one frame here, the x and y values will not change
        # and our points are in an elliptic layout with a different y value for each point,
        # we can discard points with x value less than starting point,
        # sort the rest of the inner field points by y value,
        # and append them all at once.
        # this way we get all the points from the right side of the board clockwise
        # then we can do it again for the other side
        del inner_field_cps_right[0]  # we don't want to find the starting point again
        del inner_field_cps_left[0]  # also not on the left side

        for i in get_reverse_indices(inner_field_cps_right):
            if inner_field_cps_right[i].cp.x < self.inner_fields[0].cp.x:
                del inner_field_cps_right[i]  # remove all points with x less than starting point
        [self.inner_fields.append(board_field) for board_field in inner_field_cps_right]

        for i in get_reverse_indices(inner_field_cps_left):
            if inner_field_cps_left[i].cp.x > self.inner_fields[0].cp.x:
                del inner_field_cps_left[i]  # remove all points with x more than starting point
        inner_field_cps_left = sorted(inner_field_cps_left, key=lambda board_field: board_field.isolated_cp.y,
                                      reverse=True)  # descending order
        [self.inner_fields.append(board_field) for board_field in inner_field_cps_left]

    def associate_inner_fields_with_other_fields(self):
        # for each line from inner field away from board center
        # find all closest other field center points
        # save them in a list for that inner field point
        # remove fields that were found corresponding to an inner field while doing so
        for inner_field in self.inner_fields:
            cp: Coordinate2d = inner_field.cp
            end_line_point: Coordinate2d = inner_field.extrapolated_end_point

            associated_fields = [inner_field]

            # traverse backwards so we can remove associated fields for future loops and save time doing so
            reverse_indices: list = get_reverse_indices(self.not_inner_fields)
            for i in reverse_indices:
                other_field: BoardSegment = self.not_inner_fields[i]
                other_field_point = other_field.cp
                line = [[cp.x, cp.y], [end_line_point.x, end_line_point.y]]
                cnt = np.array(line).reshape((-1, 1, 2)).astype(np.int32)  # create an opencv contour
                # pointPolygonTest returns negative distance because the point is outside the contour
                distance_from_line = np.abs(cv.pointPolygonTest(cnt, (other_field_point.x, other_field_point.y), True))
                if distance_from_line < 15:
                    associated_fields.append(other_field)
                    # other_fields_and_distances.remove(i - 1)
                    del self.not_inner_fields[i]
            self.unsorted_associated_fields.append(associated_fields)

    def sort_and_filter_associated_fields_list(self):
        for number_fields in self.unsorted_associated_fields:
            number_fields.sort(key=lambda nf: nf.isolated_distance_from_board_center)  # inplace

            if len(number_fields) > 4:  # inner, triple, outer, double
                del number_fields[4:]

            assert len(number_fields) >= 4, "Not enough board fields to approximate field borders."

            associated_fields = AssociatedBoardSegments(number_fields)
            self.associated_number_segments.append(associated_fields)


    def categorize_contours(self, detection_settings: Type[DetectionSettings]):

        self.calculate_isolated_cps()
        self.calculate_mean_isolated_board_point()

        min_iso_x = min([cnt.isolated_cp.x for cnt in self.all_contours])
        min_iso_y = min([cnt.isolated_cp.y for cnt in self.all_contours])
        max_iso_x = max([cnt.isolated_cp.x for cnt in self.all_contours])
        max_iso_y = max([cnt.isolated_cp.y for cnt in self.all_contours])
        iso_img = np.zeros((max_iso_y + 1 - min_iso_y, max_iso_x + 1 - min_iso_x, 1), np.uint8)
        for cnt in self.all_contours:
            iso_img[cnt.isolated_cp.y, cnt.isolated_cp.x] = 180
        iso_img[self.mean_isolated_board_center_point.y, self.mean_isolated_board_center_point.x] = 255

        for cnt in self.all_contours:
            cnt.calculate_isolated_distance(self.mean_isolated_board_center_point)

        sorted_by_isolated_distance = self.sorted_by_isolated_distance_from_board_center()
        # bullseye and bull field have nearly the same isolated distance
        # its necessary that we know which one is bullseye and which one is bull field
        # bullseye contour area is ca. 1/3 of bull field contour area
        bulls_eye_candidate_1 = sorted_by_isolated_distance[0]
        bulls_eye_candidate_2 = sorted_by_isolated_distance[1]
        if cv.contourArea(bulls_eye_candidate_1.contour) < cv.contourArea(bulls_eye_candidate_2.contour):
            self.bulls_eye_segment = BullsEyeSegment(bulls_eye_candidate_1.contour)
            self.bull_segment = BullSegment(bulls_eye_candidate_2.contour)
        else:
            self.bulls_eye_segment = BullsEyeSegment(bulls_eye_candidate_2.contour)
            self.bull_segment = BullSegment(bulls_eye_candidate_1.contour)

        other_fields = sorted_by_isolated_distance[2:]
        for cnt in other_fields:
            if cnt.isolated_distance_from_board_center < self.isolated_distance_center_to_inner:
                cnt.segment_type = SegmentType.INNER
                self.inner_fields.append(cnt)
            else:
                self.not_inner_fields.append(cnt)

        self.sort_inner_fields_clockwise()

        for inner_field in self.inner_fields:
            inner_field.extrapolated_end_point = get_line_extension_end_point(
                self.bulls_eye_segment.cp,
                inner_field.cp,
                4
            )

        assert len(self.inner_fields) == 20, f"There are not exactly 20 inner fields. {len(self.inner_fields)} were found."

        self.associate_inner_fields_with_other_fields()

        self.sort_and_filter_associated_fields_list()

    def calculate_point_angles_and_distances(self):
        self.bulls_eye_segment.calculate_point_angles_and_distances(self.bulls_eye_segment.cp)  # bullseye_segment
        self.bull_segment.calculate_point_angles_and_distances(self.bulls_eye_segment.cp)
        for section in self.associated_number_segments:
            section.inner_segment.calculate_point_angles_and_distances(self.bulls_eye_segment.cp)
            section.triple_segment.calculate_point_angles_and_distances(self.bulls_eye_segment.cp)
            section.outer_segment.calculate_point_angles_and_distances(self.bulls_eye_segment.cp)
            section.double_segment.calculate_point_angles_and_distances(self.bulls_eye_segment.cp)

    def get_edge_values(self):
        for section in self.associated_number_segments:
            section.inner_segment.get_edge_values()
            section.triple_segment.get_edge_values()
            section.outer_segment.get_edge_values()
            section.double_segment.get_edge_values()

    def get_numbered_segment_distance_interpolations(self, img_saved):

        self.bulls_eye_interp = self.bulls_eye_segment.calculate_interpolation(
            self.cam_resolution,
            self.bulls_eye_segment.cp
        )
        self.bull_interp = self.bull_segment.calculate_interpolation(
            self.cam_resolution,
            self.bulls_eye_segment.cp
        )

        inner_segments: List[InnerSegment] = [segment.inner_segment for segment in self.associated_number_segments]
        self.inner_interp_e_1_4, self.inner_interp_e_2_3 = get_interpolations(
            inner_segments,
            img_saved,
            self.bulls_eye_segment.cp
        )
        triple_segments: List[TripleSegment] = [segment.triple_segment for segment in self.associated_number_segments]
        self.triple_interp_e_1_4, self.triple_interp_e_2_3 = get_interpolations(
            triple_segments,
            img_saved,
            self.bulls_eye_segment.cp
        )

        outer_segments: List[OuterSegment] = [segment.outer_segment for segment in self.associated_number_segments]
        self.outer_interp_e_1_4, self.outer_interp_e_2_3 = get_interpolations(
            outer_segments,
            img_saved,
            self.bulls_eye_segment.cp
        )

        double_segments: List[DoubleSegment] = [segment.double_segment for segment in self.associated_number_segments]
        self.double_interp_e_1_4, self.double_interp_e_2_3 = get_interpolations(
            double_segments,
            img_saved,
            self.bulls_eye_segment.cp,
            True
        )

    def calculate_score_segments(self, img_height, img_width):

        # loop number section duos and calculate only corners first
        a_n_s = self.associated_number_segments
        section_pairs = [(a_n_s[i], (a_n_s[(i + 1) % len(a_n_s)])) for i in range(len(a_n_s))]
        for section_pair in section_pairs:
            f1_inner = section_pair[0].inner_segment
            f1_triple = section_pair[0].triple_segment
            f1_outer = section_pair[0].outer_segment
            f1_double = section_pair[0].double_segment
            f2_inner = section_pair[1].inner_segment
            f2_triple = section_pair[1].triple_segment
            f2_outer = section_pair[1].outer_segment
            f2_double = section_pair[1].double_segment

            f1_f2_inner_e_1_4_angle_1 = get_mid_angle(f1_inner.e_1_4.c_max_angle, f2_inner.e_1_4.c_min_angle)
            f1_f2_inner_e_1_4_angle_2 = get_mid_angle(f1_inner.e_1_4.mid_angle, f2_inner.e_1_4.mid_angle)
            f1_f2_inner_e_1_4_angle = get_mid_angle(f1_f2_inner_e_1_4_angle_1, f1_f2_inner_e_1_4_angle_2)
            bull_bulls_eye_point = get_endpoint_via_interp(
                self.bulls_eye_segment.cp,
                f1_f2_inner_e_1_4_angle,
                self.bulls_eye_interp
            )
            self.bulls_eye_segment.score_border.add_point(bull_bulls_eye_point)

            f1_f2_inner_bull_point = get_endpoint_via_interp(
                self.bulls_eye_segment.cp,
                f1_f2_inner_e_1_4_angle,
                self.bull_interp,
                # self.inner_interp_e_1_4  # introduces unnecessary noise
            )
            self.bull_segment.score_border.add_point(f1_f2_inner_bull_point)
            f1_inner.score_border.c_4 = f1_f2_inner_bull_point
            f2_inner.score_border.c_1 = f1_f2_inner_bull_point

            f1_f2_inner_e_2_3_angle1 = get_mid_angle(f1_inner.e_2_3.c_max_angle, f2_inner.e_2_3.c_min_angle)
            f1_f2_inner_e_2_3_angle2 = get_mid_angle(f1_inner.e_2_3.mid_angle, f2_inner.e_2_3.mid_angle)
            f1_f2_inner_e_2_3_angle = get_mid_angle(f1_f2_inner_e_2_3_angle1, f1_f2_inner_e_2_3_angle2)
            f1_f2_triple_e_1_4_angle1 = get_mid_angle(f1_triple.e_1_4.c_max_angle, f2_triple.e_1_4.c_min_angle)
            f1_f2_triple_e_1_4_angle2 = get_mid_angle(f1_triple.e_1_4.mid_angle, f2_triple.e_1_4.mid_angle)
            f1_f2_triple_e_1_4_angle = get_mid_angle(f1_f2_triple_e_1_4_angle1, f1_f2_triple_e_1_4_angle2)
            f1_f2_inner_triple_angle = get_mid_angle(f1_f2_inner_e_2_3_angle, f1_f2_triple_e_1_4_angle)

            f1_f2_inner_triple_point = get_endpoint_via_interp(
                self.bulls_eye_segment.cp,
                f1_f2_inner_triple_angle,
                self.inner_interp_e_2_3,
                self.triple_interp_e_1_4
            )
            f1_inner.score_border.c_3 = f1_f2_inner_triple_point
            f1_triple.score_border.c_4 = f1_f2_inner_triple_point
            f2_inner.score_border.c_2 = f1_f2_inner_triple_point
            f2_triple.score_border.c_1 = f1_f2_inner_triple_point

            f1_f2_triple_e_2_3_angle1 = get_mid_angle(f1_triple.e_2_3.c_max_angle, f2_triple.e_2_3.c_min_angle)
            f1_f2_triple_e_2_3_angle2 = get_mid_angle(f1_triple.e_2_3.mid_angle, f2_triple.e_2_3.mid_angle)
            f1_f2_triple_e_2_3_angle = get_mid_angle(f1_f2_triple_e_2_3_angle1, f1_f2_triple_e_2_3_angle2)
            f1_f2_outer_e_1_4_angle1 = get_mid_angle(f1_outer.e_1_4.c_max_angle, f2_outer.e_1_4.c_min_angle)
            f1_f2_outer_e_1_4_angle2 = get_mid_angle(f1_outer.e_1_4.mid_angle, f2_outer.e_1_4.mid_angle)
            f1_f2_outer_e_1_4_angle = get_mid_angle(f1_f2_outer_e_1_4_angle1, f1_f2_outer_e_1_4_angle2)
            f1_f2_triple_outer_angle = get_mid_angle(f1_f2_triple_e_2_3_angle, f1_f2_outer_e_1_4_angle)

            f1_f2_triple_outer_point = get_endpoint_via_interp(
                self.bulls_eye_segment.cp,
                f1_f2_triple_outer_angle,
                self.triple_interp_e_2_3,
                self.outer_interp_e_1_4
            )
            f1_triple.score_border.c_3 = f1_f2_triple_outer_point
            f1_outer.score_border.c_4 = f1_f2_triple_outer_point
            f2_triple.score_border.c_2 = f1_f2_triple_outer_point
            f2_outer.score_border.c_1 = f1_f2_triple_outer_point

            f1_f2_outer_e_2_3_angle1 = get_mid_angle(f1_outer.e_2_3.c_max_angle, f2_outer.e_2_3.c_min_angle)
            f1_f2_outer_e_2_3_angle2 = get_mid_angle(f1_outer.e_2_3.mid_angle, f2_outer.e_2_3.mid_angle)
            f1_f2_outer_e_2_3_angle = get_mid_angle(f1_f2_outer_e_2_3_angle1, f1_f2_outer_e_2_3_angle2)
            f1_f2_double_e_1_4_angle1 = get_mid_angle(f1_double.e_1_4.c_max_angle, f2_double.e_1_4.c_min_angle)
            f1_f2_double_e_1_4_angle2 = get_mid_angle(f1_double.e_1_4.mid_angle, f2_double.e_1_4.mid_angle)
            f1_f2_double_e_1_4_angle = get_mid_angle(f1_f2_double_e_1_4_angle1, f1_f2_double_e_1_4_angle2)
            f1_f2_outer_double_angle = get_mid_angle(f1_f2_outer_e_2_3_angle, f1_f2_double_e_1_4_angle)

            f1_f2_outer_double_point = get_endpoint_via_interp(
                self.bulls_eye_segment.cp,
                f1_f2_outer_double_angle,
                self.outer_interp_e_2_3,
                self.double_interp_e_1_4
            )
            f1_outer.score_border.c_3 = f1_f2_outer_double_point
            f1_double.score_border.c_4 = f1_f2_outer_double_point
            f2_outer.score_border.c_2 = f1_f2_outer_double_point
            f2_double.score_border.c_1 = f1_f2_outer_double_point

            f1_f2_double_board_angle = get_mid_angle(f1_double.e_2_3.c_max_angle, f2_double.e_2_3.c_min_angle)
            f1_f2_double_board_point = get_endpoint_via_interp(
                self.bulls_eye_segment.cp,
                f1_f2_double_board_angle,
                self.double_interp_e_2_3
            )
            f1_double.score_border.c_3 = f1_f2_double_board_point
            f2_double.score_border.c_2 = f1_f2_double_board_point

            img = np.zeros((img_height, img_width), np.uint8)
            for f in [f1_inner, f1_double, f1_triple, f1_outer, f2_inner, f2_double, f2_triple, f2_outer]:
                for p in f.contour:
                    img[p[0][1], p[0][0]] = 60
            img[bull_bulls_eye_point.y, bull_bulls_eye_point.x] = 255
            img[f1_f2_inner_bull_point.y, f1_f2_inner_bull_point.x] = 255
            img[f1_f2_inner_triple_point.y, f1_f2_inner_triple_point.x] = 255
            img[f1_f2_triple_outer_point.y, f1_f2_triple_outer_point.x] = 255
            img[f1_f2_outer_double_point.y, f1_f2_outer_double_point.x] = 255
            img[f1_f2_double_board_point.y, f1_f2_double_board_point.x] = 255
            x = 0

        # loop solo number sections
        # can only be done when all corners are calculated!

        for section in a_n_s:
            inner_c1_c4_angle_range = get_angle_range(
                section.inner_segment.score_border.c_1.angle_to_board_center,
                section.inner_segment.score_border.c_4.angle_to_board_center
            )
            inner_c1_c4_endpoint_range = get_endpoint_range_with_distance_interpolation(
                self.bulls_eye_segment.cp,
                inner_c1_c4_angle_range,
                self.bull_interp
                # self.inner_distance_interpolation_e_1_4
            )
            section.inner_segment.score_border.c_1_1_4 = inner_c1_c4_endpoint_range[1]
            section.inner_segment.score_border.c_1_4 = inner_c1_c4_endpoint_range[2]
            section.inner_segment.score_border.c_1_4_4 = inner_c1_c4_endpoint_range[3]

            inner_triple_angle_range = get_angle_range(
                section.inner_segment.score_border.c_2.angle,
                section.inner_segment.score_border.c_3.angle
            )

            inner_triple_endpoint_range = get_endpoint_range_with_distance_interpolation(
                self.bulls_eye_segment.cp,
                inner_triple_angle_range,
                self.inner_interp_e_2_3,
                self.triple_interp_e_1_4
            )
            section.inner_segment.score_border.c_2_2_3 = inner_triple_endpoint_range[1]
            section.inner_segment.score_border.c_2_3 = inner_triple_endpoint_range[2]
            section.inner_segment.score_border.c_2_3_3 = inner_triple_endpoint_range[3]
            section.triple_segment.score_border.c_1_1_4 = inner_triple_endpoint_range[1]
            section.triple_segment.score_border.c_1_4 = inner_triple_endpoint_range[2]
            section.triple_segment.score_border.c_1_4_4 = inner_triple_endpoint_range[3]

            triple_outer_angle_range = get_angle_range(
                section.triple_segment.score_border.c_2.angle,
                section.triple_segment.score_border.c_3.angle
            )

            triple_outer_endpoint_range = get_endpoint_range_with_distance_interpolation(
                self.bulls_eye_segment.cp,
                triple_outer_angle_range,
                self.triple_interp_e_2_3,
                self.outer_interp_e_1_4
            )
            section.triple_segment.score_border.c_2_2_3 = triple_outer_endpoint_range[1]
            section.triple_segment.score_border.c_2_3 = triple_outer_endpoint_range[2]
            section.triple_segment.score_border.c_2_3_3 = triple_outer_endpoint_range[3]
            section.outer_segment.score_border.c_1_1_4 = triple_outer_endpoint_range[1]
            section.outer_segment.score_border.c_1_4 = triple_outer_endpoint_range[2]
            section.outer_segment.score_border.c_1_4_4 = triple_outer_endpoint_range[3]

            outer_double_angle_range = get_angle_range(
                section.outer_segment.score_border.c_2.angle,
                section.outer_segment.score_border.c_3.angle
            )

            outer_double_endpoint_range = get_endpoint_range_with_distance_interpolation(
                self.bulls_eye_segment.cp,
                outer_double_angle_range,
                self.outer_interp_e_2_3,
                self.double_interp_e_1_4,
            )
            section.outer_segment.score_border.c_2_2_3 = outer_double_endpoint_range[1]
            section.outer_segment.score_border.c_2_3 = outer_double_endpoint_range[2]
            section.outer_segment.score_border.c_2_3_3 = outer_double_endpoint_range[3]
            section.double_segment.score_border.c_1_1_4 = outer_double_endpoint_range[1]
            section.double_segment.score_border.c_1_4 = outer_double_endpoint_range[2]
            section.double_segment.score_border.c_1_4_4 = outer_double_endpoint_range[3]

            double_board_angle_range = get_angle_range(
                section.double_segment.score_border.c_2.angle,
                section.double_segment.score_border.c_3.angle
            )

            double_board_endpoint_range = get_endpoint_range_with_distance_interpolation(
                self.bulls_eye_segment.cp,
                double_board_angle_range,
                self.double_interp_e_2_3
            )
            section.double_segment.score_border.c_2_2_3 = double_board_endpoint_range[1]
            section.double_segment.score_border.c_2_3 = double_board_endpoint_range[2]
            section.double_segment.score_border.c_2_3_3 = double_board_endpoint_range[3]

    def associate_numbers_with_segments(self, img, img_height, img_width):
        img_mask_thresh_contour = []

        for section in self.associated_number_segments:
            double_min_point = section.double_segment.score_border.c_2
            double_max_point = section.double_segment.score_border.c_3
            behind_min_point = get_line_extension_end_point(self.bulls_eye_segment.cp, double_min_point, 1.38)
            behind_max_point = get_line_extension_end_point(self.bulls_eye_segment.cp, double_max_point, 1.38)
            if not \
                    0 < behind_min_point.x < self.cam_resolution[1] or not \
                    0 < behind_min_point.y < self.cam_resolution[0] or not \
                    0 < behind_max_point.x < self.cam_resolution[1] or not \
                    0 < behind_max_point.y < self.cam_resolution[0]:
                img_mask_thresh_contour.append([None, None, None])
            else:
                # draw_line(img, double_min_point, behind_min_point, thickness=1)
                # draw_line(img, behind_min_point, behind_max_point, thickness=1)
                # draw_line(img, behind_max_point, double_max_point, thickness=1)
                # draw_line(img, double_max_point, double_min_point, thickness=1)

                pts = np.array([[p.x, p.y] for p in [double_min_point, behind_min_point, behind_max_point, double_max_point]])
                img_copy = copy.deepcopy(img)
                mask = np.zeros_like(img)
                cv.fillPoly(mask, [pts], (255, 255, 255))
                masked_img = cv.bitwise_and(img_copy, mask)
                # hsv_mask_silver = HSVMask(LightningCondition.WHITE, 70, 179, 0, 30, 100, 200)
                # mask_silver = get_masked_img_by_hsv_values(hsv, hsv_mask_silver)
                # mask_silver = get_masked_img_silver(hsv)

                # img_number = perspective_transformation(
                #     img,
                #     behind_min_point,
                #     behind_max_point,
                #     double_max_point,
                #     double_min_point
                # )
                thresh_img, number_contour = get_number_contour(
                    masked_img,
                    img_height,
                    img_width
                )
                img_mask_thresh_contour.append([masked_img, thresh_img, number_contour])
                # image_number_contours.append(number_contour)

        live_contours = [LetterToDetermine(mtc[1], mtc[2]) for mtc in img_mask_thresh_contour]
        letter_sequence = match_letter_contours(live_contours)
        for (section, number) in zip(self.associated_number_segments, letter_sequence):
            section.associated_number = number
            for bs in section.get_sections():
                bs.associated_number = number

    def get_score_and_confidence(self, impact_point: Coordinate2d) -> (int, int):
        bs_ip_distances = []  # distances from board segment score contours to impact point
        for section in self.associated_number_segments:
            for bs in section.get_sections():
                bs_ip_distances.append([get_contour_distance_to(bs.score_border.get_points_as_np(), impact_point), bs])
        for bs in [self.bulls_eye_segment, self.bull_segment]:
            bs_ip_distances.append([get_contour_distance_to(bs.score_border.get_points(), impact_point), bs])

        bs_ip_distances.sort(key=lambda d: d[0], reverse=True)
        bs_dists_filtered = [bscd for bscd in bs_ip_distances if bscd[0] >= 0]

        bs_dists_filtered.sort(key=lambda d: d[0], reverse=True)

        # impact point cases:
        # - normal: clearly in one board segment
        # - edge: on the transition of two board segments
        # - bull/bullseye: the bull score shape includes the bullseye, so the bullseye shape wins in this case.
        # - in no score contour at all, no points
        # in either case, the relevant board segment contour list should not have more than two elements

        if len(bs_dists_filtered) == 0:  # impact point is not in a board segment
            score = 0
            confidence = 100 if bs_ip_distances[0][0] <= 0.1 else 80
        elif len(bs_dists_filtered) == 1:
            bs = bs_dists_filtered[0][1]
            score = bs.associated_number * bs.score_multiplication
            confidence = 100 if bs_dists_filtered[0][0] >= 0.1 else 80  # should be improved at some time
        elif len(bs_dists_filtered) == 2:
            bs1 = bs_dists_filtered[0][1]
            bs2 = bs_dists_filtered[1][1]
            if bs_dists_filtered[0][0] == 0 and bs_dists_filtered[1][0] == 0:  # edge
                score = bs1.associated_number * bs1.score_multiplication  # we just give back the first one
                confidence = 0  # this is not "it's 50% correct", it's about confidence and there is no confidence here
            elif issubclass(bs1, BullBullsEyeSegment) and issubclass(bs2, BullBullsEyeSegment):
                bs = bs1 if isinstance(bs1, BullsEyeSegment) else bs2
                score = bs.associated_number * bs.score_multiplication
                confidence = 100
            else:
                # if the score segment bounds are correctly coded, this case can not happen.
                raise AssertionError("Score segment bounds are not correctly coded")
        else:
            raise AssertionError("Too many board segments found as score candidates")

        return score, confidence


