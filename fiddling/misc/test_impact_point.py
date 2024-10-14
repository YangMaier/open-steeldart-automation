import pathlib
import cv2 as cv

from operations_on.contours import get_impact_point


img_empty_board_path = pathlib.Path().absolute().joinpath("../media/fiddling/empty_board_1080p_distance_mid.png")
# img_one_dart_path = pathlib.Path().absolute().joinpath("../media/fiddling/dart_in_white_field.png")
# img_one_dart_path = pathlib.Path().absolute().joinpath("../media/fiddling/dart_in_20_1080p_distance_mid.png")
img_one_dart_path = pathlib.Path().absolute().joinpath("../media/fiddling/dart_in_12_1080p_distance_mid.png")
# img_one_dart_path = pathlib.Path().absolute().joinpath("../media/fiddling/dart_in_20_1080p_distance_mid.png")

img_empty_board = cv.imread(str(img_empty_board_path))
img_one_dart = cv.imread(str(img_one_dart_path))

impact_point = get_impact_point(img_empty_board, img_one_dart)

x = 0
