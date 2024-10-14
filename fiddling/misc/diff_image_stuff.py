import cv2 as cv
import numpy as np
import pathlib
import copy

from operations_on.images import skikit_similarity
from fiddling.misc import sort_points_clockwise

image1_saved = cv.imread(str(pathlib.Path().absolute().joinpath("../media/saved/impact_behind_dart_scheme_complete/sample_video_cap_07.png")))
image2_saved = cv.imread(str(pathlib.Path().absolute().joinpath("../media/saved/impact_behind_dart_scheme_complete/sample_video_cap_08.png")))

image1 = copy.deepcopy(image1_saved)
image2 = copy.deepcopy(image2_saved)

skikit_similarity(image1, image2)

difference = cv.absdiff(image1, image2)

grey = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)

thresh_diff = cv.threshold(grey, 8, 255, cv.THRESH_BINARY)[1]

contours, hierarchy = cv.findContours(thresh_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

all_contours_points_list = []
for i in range(0, len(contours)):
    for p in contours[i]:
        all_contours_points_list.append(p[0])

center_pt = np.array(all_contours_points_list).mean(axis=0)

clock_ang_dist = sort_points_clockwise.ClockwiseAngleAndDistance(center_pt)

all_contours_clockwise = sorted(all_contours_points_list, key=clock_ang_dist)

all_contours_points_clockwise_cv = (np.array(all_contours_clockwise).reshape((-1,1,2)).astype(np.int32))

mask = np.zeros(grey.shape[:2], dtype="uint8")
cv.drawContours(mask, [all_contours_points_clockwise_cv], -1, 255, -1)
masked_frame = cv.bitwise_and(image2, image2, mask=mask)

x = 0


# drawn_img = np.zeros(image2.shape[:2], dtype="uint8")
# cv.drawContours(drawn_img, contours, -1, 255, 1)
# cv.drawContours(drawn_img, all_contours_points_clockwise_cv, -1, 255, 1)

# x,y,w,h = cv.boundingRect(all_contours_points_clockwise_cv)
# cv.rectangle(drawn_img,(x,y),(x+w,y+h),255,1)
# bottom_left = (x, y+h)
# bottom_right = (x+w, y+h)


# convex_hull = cv.convexHull(all_contours_points_clockwise_cv)

# epsilon = 1 / 100 * cv.arcLength(all_contours_points_clockwise_cv, True)
# approx = cv.approxPolyDP(all_contours_points_clockwise_cv, epsilon, True)

# rect = cv.minAreaRect(all_contours_points_clockwise_cv)
# box = cv.boxPoints(rect)
# box = np.int0(box)
# box_img = cv.drawContours(drawn_img,[box],0,128,1)
#
# rows,cols = drawn_img.shape[:2]
# [vx,vy,x,y] = cv.fitLine(all_contours_points_clockwise_cv, cv.DIST_L2,0,0.01,0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols-x)*vy/vx)+y)
# cv.line(drawn_img,(cols-1,righty),(0,lefty),255,1)
#
# kernel = np.ones((15, 15), np.uint8)
# tophat = cv.morphologyEx(thresh_diff, cv.MORPH_TOPHAT, kernel)
#
# small_contours = []
# for cnt in contours:
#     if cv.contourArea(cnt) < 100:
#         small_contours.append(cnt)
#
# empty_img = np.zeros(image2.shape[:2], dtype="uint8")
# cv.drawContours(empty_img, small_contours, -1, 255, -1)
#
# x = 0