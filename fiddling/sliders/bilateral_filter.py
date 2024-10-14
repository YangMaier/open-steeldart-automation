import os
import pathlib

import cv2 as cv

def nothing(x):
    pass


def canny_sliders():
    cam_id = 0
    cam = cv.VideoCapture(cam_id)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    # cam.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    # cam.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    frame_path = pathlib.Path().absolute().joinpath("../media/fiddling")
    os.makedirs(frame_path, exist_ok=True)
    frame_basename = 'sample_video_cap'
    base_path = os.path.join(frame_path, frame_basename)
    frame_num = 0

    digit = len(str(int(cam.get(cv.CAP_PROP_FRAME_COUNT))))

    # cv.namedWindow(preview_name)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    cv.namedWindow('blurred')
    # Diameter of each pixel neighborhood.
    cv.createTrackbar('diameter', 'blurred', 20, 50, nothing)
    # The greater the value, the colors farther to each other will start to get mixed.
    cv.createTrackbar('sigmaColor', 'blurred', 75, 500, nothing)
    # The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range.
    cv.createTrackbar('sigmaSpace', 'blurred', 75, 500, nothing)

    # 30, 65, 60 bei 640x480
    # the same values look good on 1920x1080

    cv.setTrackbarPos('diameter', 'blurred', 30)
    cv.setTrackbarPos('sigmaColor', 'blurred', 65)
    cv.setTrackbarPos('sigmaSpace', 'blurred', 60)

    # best values on equalized_bgr:
    # 19, 36, 31

    while rval:
        rval, frame = cam.read()

        ycrcb_img = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(ycrcb_img)
        y_equalized = cv.equalizeHist(y)
        ycrcb = cv.merge((y_equalized, cr, cb))
        equalized_bgr = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

        # grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        diameter = cv.getTrackbarPos('diameter', 'blurred')
        if diameter % 2 == 0:
            diameter += 1

        sigmaColor = cv.getTrackbarPos('sigmaColor', 'blurred')
        sigmaSpace = cv.getTrackbarPos('sigmaSpace', 'blurred')

        img_bilateral_filter = cv.bilateralFilter(equalized_bgr, diameter, sigmaColor, sigmaSpace)

        cv.imshow("img_bilateral_filter", img_bilateral_filter)
        cv.imshow("frame", frame)

        key = cv.waitKey(20)
        if key == ord('c'):
            cv.imwrite('{}_{}.{}'.format(base_path, str(frame_num).zfill(digit), 'png'), frame)
            print("Screenshot saved!")
            frame_num += 1
        if key == 27:  # exit on ESC
            cam.release()
            cv.destroyAllWindows()
            break


canny_sliders()


