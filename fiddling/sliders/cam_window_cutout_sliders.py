
import cv2 as cv


def nothing(x):
    pass


def calibration_sliders(preview_name, cam_id):
    cam = cv.VideoCapture(cam_id)
    # cv.namedWindow(preview_name)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    cv.namedWindow(preview_name)
    cv.createTrackbar('x', preview_name, 0, 1000, nothing)
    cv.createTrackbar('y', preview_name, 0, 1000, nothing)
    cv.createTrackbar('w', preview_name, 0, 1000, nothing)
    cv.createTrackbar('h', preview_name, 0, 1000, nothing)
    cv.setTrackbarPos('x', preview_name, 0)
    cv.setTrackbarPos('y', preview_name, 0)
    cv.setTrackbarPos('w', preview_name, 500)
    cv.setTrackbarPos('h', preview_name, 500)

    while rval:

        rval, frame = cam.read()

        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        x = cv.getTrackbarPos('x', preview_name)
        y = cv.getTrackbarPos('y', preview_name)
        w = cv.getTrackbarPos('w', preview_name)
        h = cv.getTrackbarPos('h', preview_name)

        dst = grey[y:y + h, x:x + w]

        cv.imshow(preview_name, dst)
        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            cam.release()
            cv.destroyAllWindows()
            break


calibration_sliders('calibration sliders', 0)
