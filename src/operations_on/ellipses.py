import cv2 as cv
import numpy as np


from src.data_structures.ellipse import Ellipse


def get_fit_ellipse(contour, img_width, img_height):
    contour = np.asarray(contour, dtype=np.int32)
    ellipse_values = cv.fitEllipse(contour)
    img = np.zeros([img_height, img_width, 1], dtype="uint8")
    cv.ellipse(img, ellipse_values, (255, 255, 255), -1)
    processed_ellipse_contours, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    return processed_ellipse_contours[0]


def get_mean_ellipse(ellipses):
    # currently not in use
    return Ellipse(
        int(np.mean([ellipse.x for ellipse in ellipses])),
        int(np.mean([ellipse.y for ellipse in ellipses])),
        int(np.mean([ellipse.a for ellipse in ellipses])),
        int(np.mean([ellipse.b for ellipse in ellipses])),
        np.mean([ellipse.angle for ellipse in ellipses]))
