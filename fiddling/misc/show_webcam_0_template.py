import os
import pathlib

import cv2 as cv
import numpy as np


def webcam_template():
    cam_id = 1
    cam = cv.VideoCapture(cam_id)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    # frame_path = pathlib.Path().absolute().joinpath("../../src/tests/empty_board_calibration/empty_boards")
    frame_path = pathlib.Path().absolute().joinpath("../../fiddling/dataset_preparation/dart_cutouts_and_masks/todo")
    os.makedirs(frame_path, exist_ok=True)
    frame_basename = f'cam_{cam_id}'
    base_path = os.path.join(frame_path, frame_basename)
    frame_num = 0

    digit = len(str(int(cam.get(cv.CAP_PROP_FRAME_COUNT))))

    # cv.namedWindow(preview_name)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    while rval:
        rval, frame = cam.read()

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

    cam.release()

webcam_template()
