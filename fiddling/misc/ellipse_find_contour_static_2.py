from __future__ import print_function

import pathlib

import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)


def thresh_callback(val):
    threshold = val

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    canny_thicker = cv.dilate(canny_output, kernel)

    contours, _ = cv.findContours(canny_thicker, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find the rotated rectangles and ellipses for each contour
    minRect = [None] * len(contours)
    minEllipse = [None] * len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv.minAreaRect(c)
    if c.shape[0] > 5:
        minEllipse[i] = cv.fitEllipse(c)

    # Draw contours + rotated rects + ellipses

    drawing = np.zeros((canny_thicker.shape[0], canny_thicker.shape[1], 3), dtype=np.uint8)

    for i, c in enumerate(contours):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    # contour
    cv.drawContours(drawing, contours, i, color)
    # ellipse
    if c.shape[0] > 5:
        cv.ellipse(drawing, minEllipse[i], color, 2)
    # rotated rectangle
    box = cv.boxPoints(minRect[i])
    box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    cv.drawContours(drawing, [box], 0, color)

    cv.imshow('Contours', drawing)


# parser = argparse.ArgumentParser(
#     description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
# args = parser.parse_args()

# Load picture, convert to grayscale and detect edges
image_path = pathlib.Path().absolute().joinpath("../media/fiddling/cam_input.jpg")
src = cv.imread(str(image_path))

# src = cv.imread(cv.samples.findFile(args.input))
# if src is None:
#     print('Could not open or find the image:', args.input)
#     exit(0)

# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))

source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)

max_thresh = 255
thresh = 100  # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv.waitKey()
