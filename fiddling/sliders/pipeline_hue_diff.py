import cv2
import numpy as np
from skimage.metrics import structural_similarity

from operations_on.images import skikit_diff_dart_approx


def nothing(x):
    pass


def main():
    # Create a window and resize it
    cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Settings', 600, 400)  # Adjust the size as needed

    cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
    # Diameter of each pixel neighborhood.
    cv2.createTrackbar('diameter', 'Settings', 20, 50, nothing)
    # The greater the value, the colors farther to each other will start to get mixed.
    cv2.createTrackbar('sigmaColor', 'Settings', 75, 500, nothing)
    # The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range.
    cv2.createTrackbar('sigmaSpace', 'Settings', 75, 500, nothing)
    cv2.createTrackbar('value_set_hue', 'Settings', 100, 179, nothing)

    # 30, 65, 60 bei 640x480
    # the same values look good on 1920x1080

    cv2.setTrackbarPos('diameter', 'Settings', 30)
    cv2.setTrackbarPos('sigmaColor', 'Settings', 65)
    cv2.setTrackbarPos('sigmaSpace', 'Settings', 60)
    cv2.setTrackbarPos('value_set_hue', 'Settings', 170)

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    # Optionally, increase resolution for better detection of smaller items
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Adjust resolution as needed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open video stream from webcam.")
        return

    # Read a few frames to get the background image
    for i in range(30):
        ret, background = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            return

    #blur = cv2.bilateralFilter(background, 20, 30, 30)
    #b, g, r = cv2.split(blur)
    #bgr_background  = b / 4 + g / 4 + r / 4
    blur = cv2.GaussianBlur(background, (5, 5), 0)
    background_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h_background, s_background, v_background = cv2.split(background_hsv)
    # hsv_background = h_background / 4 + s_background / 4 + v_background / 4

    i = 0

    while True:
        i += 1
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        diameter = cv2.getTrackbarPos('diameter', 'Settings')
        if diameter % 2 == 0:
            diameter += 1

        sigmaColor = cv2.getTrackbarPos('sigmaColor', 'Settings')
        sigmaSpace = cv2.getTrackbarPos('sigmaSpace', 'Settings')
        value_set_hue = cv2.getTrackbarPos('value_set_hue', 'Settings')

        if i % 30 == 0:
            # blur = cv2.bilateralFilter(frame, diameter, sigmaColor, sigmaSpace)
            # b, g, r = cv2.split(blur)
            # bgr_background = b / 4 + g / 4 + r / 4
            blur = cv2.GaussianBlur(frame, (5, 5), 0)
            background_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            h_background, s_background, v_background = cv2.split(background_hsv)
            h_background[v_background < value_set_hue] = 0
            h_background[h_background > 175] = 0
            # hsv_background = h_background / 4 + s_background / 4 + v_background / 4

        #blur = cv2.bilateralFilter(frame, diameter, sigmaColor, sigmaSpace)
        #b, g, r = cv2.split(blur)
        #bgr_live = b / 4 + g / 4 + r / 4
        #cv2.imshow('Blur', blur)

        # Convert the frame to HSV color space
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        h_live, s_live, v_live = cv2.split(hsv)
        # hsv_live = h_live / 4 + s_live / 4 + v_live / 4
        # (score, diff) = structural_similarity(h_background, h_live, full=True)
        #diff = (diff * 255).astype("uint8")
        # skikit_diff = cv2.bitwise_not(diff)
        # cv2.putText(skikit_diff, str(i), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        # cv2.imshow("skikit_diff", skikit_diff)
        hue_diff = cv2.absdiff(h_live, h_background)

        hue_diff[v_live < value_set_hue] = 0
        hue_diff[hue_diff > 175] = 0
        cv2.imshow('Hue Diff', hue_diff)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()