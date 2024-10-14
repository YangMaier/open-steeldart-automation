import abc
import pathlib
from dataclasses import dataclass
from typing import List

import numpy as np

import cv2 as cv

from data_structures.colors import DartBoardColors
from data_structures.coordinates import Coordinate2dNormalized
from data_structures.score_segments import SegmentType
from operations_on.contours import filter_center_inside_contour, get_elongation_ratio, get_contour_solidity, \
    get_contour_rotated_extent
import personal_settings


@dataclass
class SegmentTemplateShort(abc.ABC):
    contour: np.ndarray
    cp_normalized: Coordinate2dNormalized
    area_normalized: float
    matched: bool = False

    def is_similarly_elongated(self, live_contour: np.ndarray, scale_elongation_min, scale_elongation_max) -> bool:
        # Elongation is the ratio of the rotated rectangles width and height
        own_elongation = get_elongation_ratio(self.contour)
        min_elongation = own_elongation * scale_elongation_min
        max_elongation = own_elongation * scale_elongation_max
        live_elongation = get_elongation_ratio(live_contour)
        filter_elongation = min_elongation <= live_elongation <= max_elongation
        return filter_elongation

    def is_similarly_solid(self, live_segment: 'SegmentTemplateShort', scale_solidity_min) -> bool:
        # Solidity is the ratio of contour area to its convex hull area.
        own_solidity = get_contour_solidity(self.contour)
        min_solidity = own_solidity * scale_solidity_min
        # more solid is always better
        live_solidity = get_contour_solidity(live_segment.contour)
        return min_solidity <= live_solidity

    def is_similarly_big(self, live_segment: 'SegmentTemplateShort', scale_area_min, scale_area_max):
        # Has a similar contour area size
        own_area = self.area_normalized
        min_area = own_area * scale_area_min
        max_area = own_area * scale_area_max
        live_area = live_segment.area_normalized
        return min_area <= live_area <= max_area

    def has_similar_rotated_extent(self, live_contour):
        # The ratio of contour area to rotated bounding rectangle area.
        own_rotated_extent = get_contour_rotated_extent(self.contour)
        rot_extent_min = own_rotated_extent * 0.8
        rot_extent_max = own_rotated_extent * 1.2
        live_cnt_rotated_extent = get_contour_rotated_extent(live_contour)

        return rot_extent_min < live_cnt_rotated_extent < rot_extent_max

    def has_expected_contour_properties(self, live_segment: 'SegmentTemplateShort', scale_elongation_min,
            scale_elongation_max,
            scale_solidity_min,
            scale_area_min,
            scale_area_max,
            segment_type):
        f1 = self.is_similarly_elongated(live_segment.contour, scale_elongation_min, scale_elongation_max)
        f2 = self.is_similarly_big(live_segment, scale_area_min, scale_area_max)
        f3 = self.is_similarly_solid(live_segment, scale_solidity_min)
        f4 = filter_center_inside_contour(live_segment.contour)
        f5 = self.has_similar_rotated_extent(live_segment.contour)
        if segment_type == SegmentType.BULLS_EYE:
            return True, f2, True, f4, True
        else:
            return f1, f2, f3, f4, f5


@dataclass
class FoundSegments:
    bulls_eye_contours: List[np.ndarray]
    bull_contours: List[np.ndarray]
    inner_contours: List[np.ndarray]
    triple_contours: List[np.ndarray]
    outer_contours: List[np.ndarray]
    double_contours: List[np.ndarray]

    def get_segments_as_masks(self, img_width, img_height):
        masks = []
        for segment_type_contours in [self.bulls_eye_contours, self.bull_contours, self.inner_contours, self.triple_contours, self.outer_contours, self.double_contours]:
            mask = np.zeros((img_height, img_width), np.uint8)
            cv.drawContours(mask, segment_type_contours, -1, (255, 255, 255), -1)
            masks.append(mask)

        return masks

    def get_as_mask(self, img_width, img_height):
        all_contours_mask = np.zeros((img_height, img_width), np.uint8)
        masks = self.get_segments_as_masks(img_width, img_height)
        for mask in masks:
            all_contours_mask = cv.bitwise_or(all_contours_mask, mask)

        return all_contours_mask


@dataclass
class SegmentMatchShort:
    contour: np.ndarray
    segment_type: SegmentType

@dataclass
class SegmentTemplateMatch:
    pre_saved_template: SegmentTemplateShort
    live_template: SegmentTemplateShort
    img_diff: np.ndarray
    scale_change: float
    rotation_change: float  # radians, 0 to 2*pi
    match_value: float
    cp_distance: float  # 0 to 1
    original_area_diff: float


class ExpectedTemplates:

    def __init__(self, template_folder_path: pathlib.Path, expected_matches: int, segment_type: SegmentType):
        self.expected_matches: int = expected_matches
        self.segment_type = segment_type

        self.segment_templates: List[SegmentTemplateShort] = []

        for file_path in template_folder_path.iterdir():
            file = cv.imread(str(file_path))
            filename = file_path.name
            # expected filename format: <'contour'>_<x_normalized>_<_y_normalized>_<cnt_area_normalized>'.png
            split_filename = filename.split("_")
            contour_cp = Coordinate2dNormalized(float(split_filename[1]), float(split_filename[2]))
            cnt_area_norm = float(split_filename[3].replace(".png", ""))
            grey = cv.cvtColor(file, cv.COLOR_BGR2GRAY)
            # thresh = cv.threshold(grey, 0, 255, cv.THRESH_BINARY)[1]
            thresh_resized = cv.resize(
                grey,
                None,
                fx=personal_settings.DetectionSettings1080p.template_matching_resolution_scale,
                fy=personal_settings.DetectionSettings1080p.template_matching_resolution_scale
            )
            contour, _ = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contour_sizes = list(map(cv.contourArea, contour))
            biggest_contour = contour[np.argmax(contour_sizes)]
            rect = cv.minAreaRect(biggest_contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            self.segment_templates.append(
                SegmentTemplateShort(
                    biggest_contour,
                    contour_cp,
                    cnt_area_norm
                )
            )

    def get_min_and_max_area_normalized(self) -> (float, float):
        min_area = 50_000
        max_area = 0
        for segment_template in self.segment_templates:
            if segment_template.area_normalized < min_area:
                min_area = segment_template.area_normalized
            if segment_template.area_normalized > max_area:
                max_area = segment_template.area_normalized
        return min_area, max_area


@dataclass
class ExpectedTemplateTypes(abc.ABC):
    path_templates_bulls_eye: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath("media/contours/bulls_eye")
    path_templates_bull: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath("media/contours/bull")
    path_templates_inner: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath("media/contours/inner")
    path_templates_triples: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath("media/contours/triple")
    path_templates_outer: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath("media/contours/outer")
    path_templates_doubles: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath("media/contours/double")
    bulls_eye_matches: ExpectedTemplates = ExpectedTemplates(path_templates_bulls_eye, 0, SegmentType.BULLS_EYE)
    bull_matches: ExpectedTemplates = ExpectedTemplates(path_templates_bull, 0, SegmentType.BULL)
    inner_matches: ExpectedTemplates = ExpectedTemplates(path_templates_inner, 0, SegmentType.INNER)
    triple_matches: ExpectedTemplates = ExpectedTemplates(path_templates_triples, 0, SegmentType.TRIPLE)
    outer_matches: ExpectedTemplates = ExpectedTemplates(path_templates_outer, 0, SegmentType.OUTER)
    double_matches: ExpectedTemplates = ExpectedTemplates(path_templates_doubles, 0, SegmentType.DOUBLE)

    def get_relevant_template_types(self) -> List[ExpectedTemplates]:
        template_matches = [
            self.bulls_eye_matches,
            self.bull_matches,
            self.inner_matches,
            self.triple_matches,
            self.outer_matches,
            self.double_matches
        ]

        return [tm for tm in template_matches if tm.expected_matches > 0]

    def get_relevant_segment_sizes(self):
        min_area = 50_000
        max_area = 0
        for segment_template in self.get_relevant_template_types():
            st_min_area, st_max_area = segment_template.get_min_and_max_area_normalized()
            if st_min_area < min_area:
                min_area = st_min_area
            if st_max_area > max_area:
                max_area = st_max_area
        return min_area, max_area


@dataclass
class ExpectedTemplatesRed(ExpectedTemplateTypes):

    def __post_init__(self):
        self.color = DartBoardColors.RED
        self.bulls_eye_matches.expected_matches = 1
        self.bull_matches.expected_matches = 0
        self.inner_matches.expected_matches = 0
        self.triple_matches.expected_matches = 10
        self.outer_matches.expected_matches = 0
        self.double_matches.expected_matches = 10


@dataclass
class ExpectedTemplatesGreen(ExpectedTemplateTypes):

    def __post_init__(self):
        self.color = DartBoardColors.GREEN
        self.bulls_eye_matches.expected_matches = 1
        self.bull_matches.expected_matches = 1
        self.inner_matches.expected_matches = 0
        self.triple_matches.expected_matches = 10
        self.outer_matches.expected_matches = 0
        self.double_matches.expected_matches = 10


@dataclass
class ExpectedTemplatesWhite(ExpectedTemplateTypes):

    def __post_init__(self):
        self.color = DartBoardColors.WHITE
        self.bulls_eye_matches.expected_matches = 0
        self.bull_matches.expected_matches = 0
        self.inner_matches.expected_matches = 10
        self.triple_matches.expected_matches = 0
        self.outer_matches.expected_matches = 10
        self.double_matches.expected_matches = 0


@dataclass
class ExpectedTemplatesBlack(ExpectedTemplateTypes):

    def __post_init__(self):
        self.color = DartBoardColors.BLACK
        self.bulls_eye_matches.expected_matches = 0
        self.bull_matches.expected_matches = 0
        self.inner_matches.expected_matches = 10
        self.triple_matches.expected_matches = 0
        self.outer_matches.expected_matches = 10
        self.double_matches.expected_matches = 0
