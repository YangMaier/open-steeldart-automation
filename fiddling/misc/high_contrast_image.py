import cv2 as cv
import numpy as np


def high_contrast_image(gray_image):
    xp = [0, 64, 112, 128, 144, 192, 255]  # setting reference values
    fp = [0, 16, 64, 128, 192, 240, 255]  # setting values to be taken for reference values
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')  # creating lookup table
    img = cv.LUT(gray_image, table)  # changing values based on lookup table

    return img

