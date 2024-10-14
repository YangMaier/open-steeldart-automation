import pathlib

import cv2 as cv
import numpy as np

img_path = pathlib.Path().absolute().joinpath("dart_1_cutout.png")
img = cv.imread(str(img_path))
grey_img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
contours, _ = cv.findContours(grey_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
biggest_contour = max(contours, key=cv.contourArea)
new_img = np.zeros_like(grey_img)
cv.drawContours(new_img, [biggest_contour], -1, (255, 255, 255), -1)
split_name = img_path.name.split("_")
mask_name = split_name[0] + "_" + split_name[1] + "_mask.png"
cv.imwrite(str(img_path.parent.joinpath(mask_name)), new_img)
