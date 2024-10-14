import cv2 as cv
import sys
import numpy as np

def nothing(x):
    pass

# Create a window
cv.namedWindow('image', cv.WINDOW_NORMAL)

# create trackbars for color change
cv.createTrackbar('HMin', 'image', 0, 179, nothing)  # Hue is from 0-179 for Opencv
cv.createTrackbar('SMin', 'image', 0, 255, nothing)
cv.createTrackbar('VMin', 'image', 0, 255, nothing)
cv.createTrackbar('HMax', 'image', 0, 179, nothing)
cv.createTrackbar('SMax', 'image', 0, 255, nothing)
cv.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for MAX HSV trackbars.
cv.setTrackbarPos('HMax', 'image', 179)
cv.setTrackbarPos('SMax', 'image', 255)
cv.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

img = cv.imread('/media/fiddling/letter_images_old/11.png')
output = img
waitTime = 33

while True:

    # get current positions of all trackbars
    hMin = cv.getTrackbarPos('HMin','image')
    sMin = cv.getTrackbarPos('SMin','image')
    vMin = cv.getTrackbarPos('VMin','image')

    hMax = cv.getTrackbarPos('HMax','image')
    sMax = cv.getTrackbarPos('SMax','image')
    vMax = cv.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    output = cv.bitwise_and(img, img, mask=mask)

    # Print if there is a change in HSV value
    # if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
    #     print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
    #     phMin = hMin
    #     psMin = sMin
    #     pvMin = vMin
    #     phMax = hMax
    #     psMax = sMax
    #     pvMax = vMax

    # Display output image
    cv.imshow('image', output)

    # Wait longer to prevent freeze for videos.
    if cv.waitKey(waitTime) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()