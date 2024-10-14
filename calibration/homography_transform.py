import copy

# https://www.stat.cmu.edu/~ryantibs/statcomp-F15/lectures/simulation.html
# load dartboard reference picture

import cv2 as cv
import numpy as np
import pathlib

from operations_on.contours import get_center_of_contour_precise

image_path = pathlib.Path().absolute().joinpath("../../media/calibration/reference_img_black_no_letters_red_corners.png")
reference_img_bgr = cv.imread(str(image_path))

# invert colors because the dartboard is white
# reference_img_rgb = cv.bitwise_not(reference_img_rgb)

# convert to hsv
reference_img_hsv = cv.cvtColor(reference_img_bgr, cv.COLOR_BGR2HSV)

mask_red = get_masked_img_red(reference_img_hsv)

# get contours from corners
contours, _ = cv.findContours(mask_red, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

reference_cps_drawn = copy.deepcopy(reference_img_bgr)
reference_cps_drawn = cv.threshold(reference_cps_drawn, 0, 40, cv.THRESH_BINARY)[1]

cps = [get_center_of_contour_precise(contour) for contour in contours]

# for contour in contours:
#     cp = get_center_of_contour(contour)
#     reference_cps_drawn[cp.y, cp.x] = 255

x = 0
