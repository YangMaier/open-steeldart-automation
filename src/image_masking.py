import cv2 as cv
import numpy as np

from data_structures.hsv_mask_presets import HSVMask


def get_masked_img_by_hsv_values(hsv_img, mask: HSVMask):
    if mask.has_split_hue_values():
        mask1 = cv.inRange(hsv_img, np.array(mask.get_lower_1()), np.array(mask.get_upper_1()))
        mask2 = cv.inRange(hsv_img, np.array(mask.get_lower_2()), np.array(mask.get_upper_2()))
        masked_result = cv.bitwise_or(mask1, mask2)
    else:
        masked_result = cv.inRange(hsv_img, np.array(mask.get_lower_1()), np.array(mask.get_upper_1()))

    return masked_result
