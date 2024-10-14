import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)

window_name = "Camera 0 Lines"
cv.namedWindow(window_name)

if not cam.isOpened():  # try to get the first frame
    print("Cam " + str(0) + " could not be opened.")

else:
    rval, frame = cam.read()

    while rval:
        rval, frame = cam.read()
        key = cv.waitKey(20)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (3, 3), 0)

        # apply automatic Canny edge detection using the computed median
        median_with_fixed_offset = np.median(blur) + 40
        sigma = 0.33
        lower_t = int(max(0, (1.0 - sigma) * median_with_fixed_offset))
        upper_t = int(min(255, (1.0 + sigma) * median_with_fixed_offset))
        canny = cv.Canny(blur, lower_t, upper_t)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        canny_thicker = cv.dilate(canny, kernel)

        # Apply HoughLinesP method to directly obtain line end points
        # lines_list = []
        lines = cv.HoughLinesP(
            canny_thicker,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 180,  # Angle resolution in radians
            threshold=40,  # Min number of votes for valid line
            minLineLength=400,  # Min allowed length of line
            maxLineGap=20  # Max allowed gap between line for joining them
        )

        # Iterate over points
        if lines is not None:
            for points in lines:
                # Extracted points nested in the list
                x1, y1, x2, y2 = points[0]
                # Draw the lines joing the points
                # On the original image
                cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Maintain a simples lookup list for points
                # lines_list.append([(x1, y1), (x2, y2)])

        # Using cv.putText() method
        frame = cv.putText(frame, "Lower canny: " + str(lower_t) + "\nUpper canny: " + str(upper_t) + "\nMedian: " + str(median_with_fixed_offset),
                            (5, 30), cv.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1, cv.LINE_AA)
        cv.imshow(window_name, frame)
        cv.imshow("Canny", canny_thicker)
        if key == 27:  # exit on ESC
            cv.destroyWindow(window_name)
            cv.destroyWindow("Canny")
            cam.release()
            break
