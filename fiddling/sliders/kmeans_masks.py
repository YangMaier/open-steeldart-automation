import os
import pathlib
from typing import List, Type

import cv2 as cv
import numpy as np

from data_structures.score_segments import SegmentType
from operations_on.contours import get_center_of_contour_normalized, get_contour_area_normalized, \
    get_test_img_for_contour
from operations_on.coordinates import get_distance_normalized
from data_structures.segment_template import ExpectedTemplatesRed, ExpectedTemplatesGreen, \
    ExpectedTemplateTypes, SegmentTemplateShort, ExpectedTemplatesWhite, ExpectedTemplatesBlack, \
    SegmentMatchShort, FoundSegments


def get_dominant_color(pil_img):
    img = pil_img.copy()
    img = img.convert("RGBA")
    img = img.resize((1, 1), resample=0)
    dominant_color = img.getpixel((0, 0))
    return dominant_color


def nothing(x):
    pass


def match_live_segment_with_template(live_segment: SegmentTemplateShort, template_segment: SegmentTemplateShort):
    # here, the template is already in the region of the live segment
    # and the only thing we currently want to test against, is the contour size, so just compare the norm area.
    area_min = template_segment.area_normalized * 0.7
    area_max = template_segment.area_normalized * 1.3
    return area_min < live_segment.area_normalized < area_max


def get_relevant_segments(
        layers: List[np.ndarray],
        expected_templates: [Type[ExpectedTemplateTypes]],
        scale_elongation_min,
        scale_elongation_max,
        scale_solidity_min,
        scale_area_min,
        scale_area_max,
        scale_cnt_dist,
        layer_name):
    contour_template_distance_threshold = scale_cnt_dist
    img_width = layers[0].shape[1]
    img_height = layers[0].shape[0]

    found_segments = []
    live_segment_img = np.zeros_like(layers[0])

    for color_layer in layers:
        contours, _ = cv.findContours(color_layer, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        found_contours_img = np.zeros_like(color_layer)
        contours_filtered = [cnt for cnt in contours if cv.contourArea(cnt) > 10]
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
                for expected_color_template_type in expected_templates:
                    if not live_segment.matched:
                        expected_color_template_type: ExpectedTemplateTypes
                        for expected_template_type in expected_color_template_type.get_relevant_template_types():
                            current_template_type = expected_template_type.segment_type
                            if not live_segment.matched:
                                min_area_filter, max_area_filter = expected_template_type.get_min_and_max_area_normalized()
                                min_area_filter = min_area_filter * 0.8
                                max_area_filter = max_area_filter * 1.2
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
                                                # live_segment_img = np.zeros_like(color_layer)
                                                # cv.drawContours(live_segment_img, [live_segment.contour], -1,
                                                #                 (255, 255, 255), 1)
                                                # cv.drawContours(live_segment_img, [template.contour], -1, (255, 0, 255),
                                                #                 1)
                                                # cv.circle(
                                                #     live_segment_img,
                                                #     (int(template.cp_normalized.x * img_width),
                                                #      int(template.cp_normalized.y * img_height)),
                                                #     2,
                                                #     (255, 255, 255),
                                                #     1,
                                                # )
                                                # cv.putText(
                                                #     live_segment_img,
                                                #     f"type: {expected_template_type.segment_type}, t exp.: {fulfills_template_expectations}",
                                                #     (30, 30),
                                                #     cv.FONT_HERSHEY_SIMPLEX,
                                                #     0.5,
                                                #     (255, 255, 255),
                                                #     1,
                                                #     cv.LINE_AA
                                                # )
                                                break
                                            else:  # for else

                                                cv.drawContours(live_segment_img, [live_segment.contour], -1,
                                                                (255, 255, 255), 1)
                                                # cv.drawContours(live_segment_img, [template.contour], -1, (255, 0, 255), 1)
                                                # cv.circle(
                                                #     live_segment_img,
                                                #     (int(template.cp_normalized.x * img_width), int(template.cp_normalized.y * img_height)),
                                                #     2,
                                                #     (255, 255, 255),
                                                #     1,
                                                # )
                                                f1t = "t" if f1 else "f"
                                                f2t = "t" if f2 else "f"
                                                f3t = "t" if f3 else "f"
                                                f4t = "t" if f4 else "f"
                                                f5t = "t" if f5 else "f"
                                                cv.putText(
                                                    live_segment_img,
                                                    f"t: {expected_template_type.segment_type.name}, f1{f1t} f2{f2t} f3{f3t} f4{f4t} f5{f5t}",
                                                    (int(live_segment.cp_normalized.x * img_width), int(live_segment.cp_normalized.y * img_height)),
                                                    cv.FONT_HERSHEY_SIMPLEX,
                                                    0.4,
                                                    (255, 255, 255),
                                                    1,
                                                    cv.LINE_4
                                                )
        et: ExpectedTemplateTypes = expected_templates[0]
        bem = et.bulls_eye_matches.expected_matches
        bm = et.bull_matches.expected_matches
        im = et.inner_matches.expected_matches
        tm = et.triple_matches.expected_matches
        om = et.outer_matches.expected_matches
        dm = et.double_matches.expected_matches
        fbem = len([fm for fm in found_segments if fm.segment_type == SegmentType.BULLS_EYE])
        fbm = len([fm for fm in found_segments if fm.segment_type == SegmentType.BULL])
        fim = len([fm for fm in found_segments if fm.segment_type == SegmentType.INNER])
        ftm = len([fm for fm in found_segments if fm.segment_type == SegmentType.TRIPLE])
        fom = len([fm for fm in found_segments if fm.segment_type == SegmentType.OUTER])
        fdm = len([fm for fm in found_segments if fm.segment_type == SegmentType.DOUBLE])
        if bem != fbem or bm != fbm or im != fim or tm != ftm or om != fom or dm != fdm:
            cv.imshow(f"{layer_name} not matched", live_segment_img)

    contours_bulls_eye = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.BULLS_EYE]
    contours_bull = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.BULL]
    contours_inner = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.INNER]
    contours_triple = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.TRIPLE]
    contours_outer = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.OUTER]
    contours_double = [fm.contour for fm in found_segments if fm.segment_type == SegmentType.DOUBLE]
    #
    # img_bulls_eye_contours = np.zeros(layers[0].shape[:2], dtype="uint8")
    # img_bull_contours = np.zeros(layers[0].shape[:2], dtype="uint8")
    # img_inner_contours = np.zeros(layers[0].shape[:2], dtype="uint8")
    # img_triple_contours = np.zeros(layers[0].shape[:2], dtype="uint8")
    # img_outer_contours = np.zeros(layers[0].shape[:2], dtype="uint8")
    # img_double_contours = np.zeros(layers[0].shape[:2], dtype="uint8")
    #
    # cv.drawContours(img_bulls_eye_contours, contours_bulls_eye, -1, (255, 255, 255, -1))
    # cv.drawContours(img_bull_contours, contours_bull, -1, (255, 255, 255, -1))
    # cv.drawContours(img_inner_contours, contours_inner, -1, (255, 255, 255, -1))
    # cv.drawContours(img_triple_contours, contours_triple, -1, (255, 255, 255, -1))
    # cv.drawContours(img_outer_contours, contours_outer, -1, (255, 255, 255, -1))
    # cv.drawContours(img_double_contours, contours_double, -1, (255, 255, 255, -1))

    found_segments = FoundSegments(
        contours_bulls_eye, contours_bull, contours_inner, contours_triple, contours_outer, contours_double
    )

    return found_segments, live_segment_img


def canny_sliders():
    cam_id = 2
    cam = cv.VideoCapture(cam_id, cv.CAP_DSHOW)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    # cam.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    # cam.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    frame_path = pathlib.Path().absolute().joinpath("../media/fiddling")
    os.makedirs(frame_path, exist_ok=True)
    frame_basename = 'sample_video_cap'
    base_path = os.path.join(frame_path, frame_basename)
    frame_num = 0

    digit = len(str(int(cam.get(cv.CAP_PROP_FRAME_COUNT))))

    # cv.namedWindow(preview_name)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    grey_stub = np.zeros_like(frame)

    cv.namedWindow('sliders')
    cv.createTrackbar('blur', 'sliders', 0, 20, nothing)
    cv.createTrackbar('kmeans k', 'sliders', 1, 10, nothing)

    cv.setTrackbarPos('blur', 'sliders', 20)
    cv.setTrackbarPos('kmeans k', 'sliders', 7)

    cam_frames = 0
    while cam_frames < 30:
        rval, frame = cam.read()
        cam_frames += 1

    while rval:
        rval, bgr = cam.read()

        blur_val = cv.getTrackbarPos('blur', 'sliders')
        if blur_val % 2 == 0:
            blur_val += 1
        image = cv.GaussianBlur(bgr, (blur_val, blur_val), 0)
        kmeans_k = cv.getTrackbarPos('kmeans k', 'sliders')
        layers, new_img = get_kmeans_layers(image, kmeans_k)
        for i, layer in enumerate(layers):
            cv.imshow(f'layer {i}', layer)

        expected_segments_red = ExpectedTemplatesRed()
        expected_segments_green = ExpectedTemplatesGreen()
        expected_segments_white = ExpectedTemplatesWhite()
        expected_segments_black = ExpectedTemplatesBlack()
        bulls_eye_mask, bull_mask, inner_mask, triple_mask, outer_mask, double_mask = get_relevant_segments(
            layers,
            [
                expected_segments_red,
                expected_segments_green,
                expected_segments_white,
                expected_segments_black
            ],
        0.12
        )
        # kmeans_k_val = cv.getTrackbarPos('kmeans k', 'sliders')
        # segmented_image = get_kmeans(blur_bgr, kmeans_k_val)
        cv.imshow("bulls_eye_mask", bulls_eye_mask)
        cv.imshow("bull_mask", bull_mask)
        cv.imshow("inner_mask", inner_mask)
        cv.imshow("triple_mask", triple_mask)
        cv.imshow("outer_mask", outer_mask)
        cv.imshow("double_mask", double_mask)
        # cv.imshow("sharpen_inverted", sharpen_inverted)
        # cv.imshow("scharr", scharr_edges)
        cv.imshow("sliders", grey_stub)
        # cv.imshow("segmented image", segmented_image)

        cv.waitKey(0)

        key = cv.waitKey(20)
        if key == ord('c'):
            cv.imwrite('{}_{}.{}'.format(base_path, str(frame_num).zfill(digit), 'png'), bgr)
            print("Screenshot saved!")
            frame_num += 1
        if key == 27:  # exit on ESC
            cam.release()
            cv.destroyAllWindows()
            break


def get_kmeans_layers(image, num_colors):
    h, w, c = image.shape
    # reshape to 1D array
    image_2d = image.reshape(h * w, c).astype(np.float32)
    # set number of colors
    numcolors = num_colors
    numiters = 15
    epsilon = 0.85
    attempts = 10
    # do kmeans processing
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, numiters, epsilon)
    ret, labels, centers = cv.kmeans(image_2d, numcolors, None, criteria, attempts, cv.KMEANS_RANDOM_CENTERS)
    # reconstitute 2D image of results
    centers = np.uint8(centers)
    new_image = centers[labels.flatten()]
    new_image = new_image.reshape(image.shape)
    k = 0
    layers = []
    for center in centers:
        # select color and create mask
        # print(center)
        layer = new_image.copy()
        mask = cv.inRange(layer, center, center)

        # apply mask to layer
        layer[mask == 0] = [0, 0, 0]
        layers.append(layer)
        k = k + 1

    return layers, new_image
