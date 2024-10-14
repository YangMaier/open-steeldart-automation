import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random as rng

# https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/

# Load picture, convert to grayscale and detect edges
image_path = pathlib.Path().absolute().joinpath("../media/fiddling/cam_input.jpg")
image_rgb = cv2.imread(str(image_path))

# Apply canny to grayscale image. This ensures that there will be less noise during the edge detection process.
gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
# 1. Smoothing
# 2. Computing image gradients
# 3. Applying non - maxima suppression
# 4. Utilizing hysteresis thresholding
# Step 1 Smoothing: Smoothing an image allows us to ignore much of the detail and instead focus on the actual
# structure. This also makes sense in the context of edge detection — we are not interested in the actual
# detail of the image.
# Instead, we want to apply edge detection to find the structure and outline of the objects in the image,
# so we can further process them.
# (7,7) yielded the least amount of noise in canny in a direct comparison of blur filters with
# (3,3), (3,5), (5,5) and (7,7)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
# unclear: does another picture resolution result in another ksize optimum?

# canny = cv2.Canny(blur, 10, 70)
# ret, mask = cv2.threshold(canny, 120, 200, cv2.THRESH_BINARY)
# display the edge map
# cv2.imshow(preview_name, mask)

# compute a "wide", "mid-range", and "tight" threshold for the edges
# using the Canny edge detector
# wide = cv2.Canny(blur, 100, 140)
# arguments: image, lower threshold, upper threshold
# (100, 150) yielded best results with my setup
# apply automatic Canny edge detection using the computed median
sigma = 0.33
v = np.median(imCal)
#lower = int(max(0, (1.0 - sigma) * v))
#upper = int(min(255, (1.0 + sigma) * v))
edged = cv2.Canny(imCal, 250, 255)

canny = cv2.Canny(blur, 50, 120)
canny_saved = canny

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
canny_thicker = cv2.dilate(canny, kernel)
# cv2.imshow(preview_name, canny)

ellipses = cv2.ximgproc.findEllipses(
    canny_thicker,
    scoreThreshold=0.3,
    reliabilityThreshold=0.3,
    centerDistanceThreshold=0.1
)
# ellipses = cv2.ximgproc.findEllipses(canny, ellipses)

# contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours_draw = cv2.drawContours(canny, contours, -1, (255, 255, 0), 8)
#
# fig2, (ax1, ax2) = plt.subplots(
#     ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True
# )
#
# ax1.set_title('Canny')
# ax1.imshow(canny_saved)
#
# ax2.set_title('Edge (white) and result (red_transparent)')
# ax2.imshow(contours_draw)
drawing = np.zeros((canny_thicker.shape[0], canny_thicker.shape[1], 3), dtype=np.uint8)

for ellipse in ellipses:
    # random color
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv2.ellipse(drawing, (ellipse[0], ellipse[1]), (ellipse[2], ellipse[3]), color, 2)
cv2.imshow("test", drawing)

# Zwei Fenster Lösung bauen mit den drei Threshold Parametern als Input

cv2.waitKey()
# plt.show()
