
import cv2 as cv
import numpy as np


def get_kmeans(bgr, k):
    # k is number of clusters
    # Change color to RGB (from BGR)
    image = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))  # numpy reshape operation -1 unspecified
    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)
    # criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]  # Mapping labels to center points( RGB Value)
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))

    return segmented_image