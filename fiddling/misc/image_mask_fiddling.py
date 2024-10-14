import pathlib

import cv2 as cv

img_empty_board_path1 = pathlib.Path("A:/dart_motion_series/1724320550845081700_1/1724320550845081700_0_base_frame.png")
img_empty_board_path2 = pathlib.Path("A:/dart_motion_series/1724320550871626700_0/1724320550871626700_0_base_frame.png")
img_board_path_1 = pathlib.Path("A:/dart_motion_series/1724320550845081700_1/1724320551650575400_1_frame.png")
img_board_path_2 = pathlib.Path("A:/dart_motion_series/1724320550871626700_0/1724320551284536700_1_frame.png")
img_empty_board_1 = cv.imread(str(img_empty_board_path1))
img_empty_board_2 = cv.imread(str(img_empty_board_path2))
img_board_1 = cv.imread(str(img_board_path_1))
img_board_2 = cv.imread(str(img_board_path_2))

# path_template_red = pathlib.Path().absolute().joinpath("../../fiddling/media/contours/contour_0.612_0.5111 - Copy.png")
path_template_red = pathlib.Path().absolute().joinpath("../../src/media/contours/contours_red.png")
img_template_red = cv.imread(str(path_template_red))

hsv = cv.cvtColor(img_template_red, cv.COLOR_BGR2HSV)
hsv_target = cv.cvtColor(img_empty_board_1, cv.COLOR_BGR2HSV)

# calculating object histogram
M = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256])
# normalize histogram and apply backprojection
cv.normalize(M, M, 0, 255, cv.NORM_MINMAX)

B = cv.calcBackProject([hsv_target], [0, 1], M, [0, 180, 0, 256], 1)
cv.normalize(B, B, 0, 255, cv.NORM_MINMAX)
ret, thresh = cv.threshold(B, 191, 255, 0)

# Overlay images using bitwise_and
thresh = cv.merge((thresh, thresh, thresh))
res = cv.bitwise_and(img_empty_board_1, thresh)
x = 0
