from typing import List, Type

import numpy as np
import cv2 as cv

from data_structures.score_segments import SegmentType
from data_structures.segment_template import ExpectedTemplateTypes, SegmentTemplateShort, SegmentMatchShort, \
    FoundSegments
from operations_on.contours import get_center_of_contour_normalized, get_contour_area_normalized, \
    get_test_img_for_contour
from operations_on.coordinates import get_distance_normalized


def get_relevant_segments(
        grey_img: np.ndarray,
        expected_templates: ExpectedTemplateTypes,
        scale_elongation_min,
        scale_elongation_max,
        scale_solidity_min,
        scale_area_min,
        scale_area_max,
        scale_cnt_dist,
        img_width,
        img_height,
        layer_name):
    contour_template_distance_threshold = scale_cnt_dist
    area_filter_scale_min = 0.8
    area_filter_scale_max = 1.2

    found_segments = []
    not_matched_img = np.zeros((img_height, img_width), np.uint8)

    contours, _ = cv.findContours(grey_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    found_contours_img = np.zeros((img_height, img_width), np.uint8)
    min_template_area, max_template_area = expected_templates.get_relevant_segment_sizes()
    min_template_area = min_template_area * area_filter_scale_min
    max_template_area = max_template_area * area_filter_scale_max
    contours_filtered = [cnt for cnt in contours if min_template_area < get_contour_area_normalized(cnt, img_width, img_height) < max_template_area]
    cv.drawContours(found_contours_img, contours_filtered, -1, (255, 255, 255), 1)
    live_segments = []
    for contour in contours_filtered:
        cp_normalized = get_center_of_contour_normalized(contour, img_width, img_height)
        area_normalized = get_contour_area_normalized(contour, img_width, img_height)
        live_segments.append(SegmentTemplateShort(
            contour,
            cp_normalized,
            area_normalized
        ))

    for live_segment in live_segments:
        test_img_contour = get_test_img_for_contour(live_segment.contour)
        if not live_segment.matched:
            for expected_template_type in expected_templates.get_relevant_template_types():
                current_template_type = expected_template_type.segment_type
                if not live_segment.matched:
                    min_area_filter, max_area_filter = expected_template_type.get_min_and_max_area_normalized()
                    min_area_filter = min_area_filter * area_filter_scale_min
                    max_area_filter = max_area_filter * area_filter_scale_max
                    size_is_ok = min_area_filter < live_segment.area_normalized < max_area_filter
                    if not size_is_ok:
                        continue  # contour too small or too big
                    else:
                        templates_near_live_segment = [
                            pst for pst in expected_template_type.segment_templates if
                            get_distance_normalized(
                                live_segment.cp_normalized, pst.cp_normalized
                            ) < contour_template_distance_threshold
                        ]

                        for template in templates_near_live_segment:
                            if not live_segment.matched:
                                f1, f2, f3, f4, f5 = template.has_expected_contour_properties(
                                    live_segment,
                                    scale_elongation_min,
                                    scale_elongation_max,
                                    scale_solidity_min,
                                    scale_area_min,
                                    scale_area_max,
                                    expected_template_type.segment_type
                                )
                                fulfills_template_expectations = f1 and f2 and f3 and f4 and f5
                                if fulfills_template_expectations:
                                    found_segments.append(
                                        SegmentMatchShort(
                                            live_segment.contour,
                                            expected_template_type.segment_type
                                        )
                                    )
                                    live_segment.matched = True

                                    break
                                else:  # for else

                                    cv.drawContours(not_matched_img, [live_segment.contour], -1,
                                                    (255, 255, 255), 1)

                                    f1t = "t" if f1 else "f"
                                    f2t = "t" if f2 else "f"
                                    f3t = "t" if f3 else "f"
                                    f4t = "t" if f4 else "f"
                                    f5t = "t" if f5 else "f"
                                    cv.putText(
                                        not_matched_img,
                                        f"t: {expected_template_type.segment_type.name}, f1{f1t} f2{f2t} f3{f3t} f4{f4t} f5{f5t}",
                                        (int(live_segment.cp_normalized.x * img_width), int(live_segment.cp_normalized.y * img_height)),
                                        cv.FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        (255, 255, 255),
                                        1,
                                        cv.LINE_4
                                    )

        bem = expected_templates.bulls_eye_matches.expected_matches
        bm = expected_templates.bull_matches.expected_matches
        im = expected_templates.inner_matches.expected_matches
        tm = expected_templates.triple_matches.expected_matches
        om = expected_templates.outer_matches.expected_matches
        dm = expected_templates.double_matches.expected_matches
        fbem = len([fm for fm in found_segments if fm.segment_type == SegmentType.BULLS_EYE])
        fbm = len([fm for fm in found_segments if fm.segment_type == SegmentType.BULL])
        fim = len([fm for fm in found_segments if fm.segment_type == SegmentType.INNER])
        ftm = len([fm for fm in found_segments if fm.segment_type == SegmentType.TRIPLE])
        fom = len([fm for fm in found_segments if fm.segment_type == SegmentType.OUTER])
        fdm = len([fm for fm in found_segments if fm.segment_type == SegmentType.DOUBLE])

    contours_bulls_eye = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.BULLS_EYE]
    contours_bull = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.BULL]
    contours_inner = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.INNER]
    contours_triple = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.TRIPLE]
    contours_outer = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.OUTER]
    contours_double = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.DOUBLE]

    found_segments = FoundSegments(
        contours_bulls_eye, contours_bull, contours_inner, contours_triple, contours_outer, contours_double
    )

    return found_segments, not_matched_img
