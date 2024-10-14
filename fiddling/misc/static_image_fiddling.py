import pathlib
import numpy as np

from skimage import exposure

import cv2 as cv
from matplotlib import pyplot as plt

def plot_histogram(test_img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([test_img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

reference_board_img = pathlib.Path("/home/fejira/PycharmProjects/steeldart_recognition/fiddling/media/random/wheel-recreation-circle-sports-center-arrows-958826-pxhere.com.jpg")
img_empty_board_path1 = pathlib.Path("/home/fejira/PycharmProjects/steeldart_recognition/src/tests/empty_board_calibration/empty_boards/cam_0_00.png")
img_reference = cv.imread(str(reference_board_img))
img = cv.imread(str(img_empty_board_path1))

blur = cv.bilateralFilter(img,20,30,30)
# plot_histogram(blur)



ycrcb_img = cv.cvtColor(blur, cv.COLOR_BGR2YCrCb)
y, cr, cb = cv.split(ycrcb_img)
y_equalized = cv.equalizeHist(y)
ycrcb = cv.merge((y_equalized, cr, cb))
equalized_bgr = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
saturation = img_hsv[:, :, 1].mean()

multi = True if img.shape[-1] > 1 else False
matched = exposure.match_histograms(img, img_reference, channel_axis=-1)

x = 0