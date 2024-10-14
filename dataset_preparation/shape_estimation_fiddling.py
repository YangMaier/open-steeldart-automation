import pathlib

import cv2 as cv

from operations_on.images import transform_img, get_board_points
from preset_board_circle_coordinates import board_corners_reference
from personal_settings import DetectionSettings1080p
from obsolete_stuff.score_board_extraction.calculate_score_board import calculate_score_board

img_empty_board_path1 = pathlib.Path("C:/Users/Jimbob/PycharmProjects/steeldart_recognition/src/tests/empty_board_calibration/empty_boards/cam_1_02.png")
img_empty_board_path2 = pathlib.Path("C:/Users/Jimbob/PycharmProjects/steeldart_recognition/src/tests/empty_board_calibration/empty_boards/cam_0_02.png")
img_board_path_1 = pathlib.Path("A:/dart_motion_series/1724320550845081700_1/1724320551650575400_1_frame.png")
img_board_path_2 = pathlib.Path("A:/dart_motion_series/1724320550871626700_0/1724320551284536700_1_frame.png")

img_empty_board_1 = cv.imread(str(img_empty_board_path1))
img_empty_board_2 = cv.imread(str(img_empty_board_path2))
img_board_1 = cv.imread(str(img_board_path_1))
img_board_2 = cv.imread(str(img_board_path_2))

dart_board_1 = calculate_score_board(img_empty_board_1, DetectionSettings1080p)
dart_board_2 = calculate_score_board(img_empty_board_2, DetectionSettings1080p)

board_rets1 = get_board_rg_start(img_empty_board_1)

dart_board_1_corner_points = get_board_points(dart_board_1)
dart_board_2_corner_points = get_board_points(dart_board_2)

circle_board_1 = transform_img(img_board_1, dart_board_1_corner_points, board_corners_reference)
warped_board_1_to_2 = transform_img(circle_board_1, board_corners_reference, dart_board_2_corner_points)
