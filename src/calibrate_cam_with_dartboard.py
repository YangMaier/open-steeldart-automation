import pathlib

import cv2 as cv

from ultralytics import YOLO

from data_structures.score_segments import SegmentType
from fiddling.pipelines.rg_start import get_triple_and_double_contours
from operations_on.contours import get_as_board_segments


def calibrate_cam_with_dartboard(frame):
    (board_center_coordinate,
         bull_segment,
         bulls_eye_segment,
         contours_double,
         contours_triple,
         equalized_bgr,
         equalized_bgr_blurred,
         equalized_hsv_blurred,
         img_height,
         img_width) = get_triple_and_double_contours(frame)

    if len(contours_triple) == 20 and len(contours_double) == 20:
        board_segments_double = get_as_board_segments(board_center_coordinate, contours_double, SegmentType.DOUBLE)
        board_segments_double.sort(key=lambda x: x.center_cad.angle)

        path_to_calibration_model = pathlib.Path(__file__).parent.joinpath("yolo/best.pt")
        model = YOLO(str(path_to_calibration_model))
        frame_resized = cv.resize(frame, (800, 800))
        # results = model(frame, verbose=False)
        results = model.segment(frame)



        print(results)

        # YOLO boxes here
        # get box centers
        # get box center angles
        # match double segments with number boxes by angle to board center
        # calculate the numbers for all segments without a number

        # warp image code here
        # if warping does not work correctly, rotate the object points (real world) to match number of angle 0 in image
    else:
        print("Frame could not be calibrated with dartboard.")
