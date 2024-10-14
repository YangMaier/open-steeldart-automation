import os
import pathlib
import time
import keyboard

import cv2 as cv

from data_structures.cam_thread import CamThread
from operations_on.images import motion_is_detected_slow
import personal_settings


def save_motion_series(preview_name, cam_id):
    """ Saves a motion series to disk

    When motion is detected in frame flow, the motion series is saved to disk in a new sub folder

    Args:
        preview_name: window name of webcam to watch
        cam_id: system id of webcam

    Returns:

    """

    # change to your liking
    disk_path = personal_settings.motion_series_disk_path

    # maybe you need a different backend
    # https://stackoverflow.com/questions/73789013/python-cv2-changing-camera-resolution
    # print(cv2.getBuildInformation()) is your friend, look at Chapter "Video I/O" and test the available options
    # maybe try cv.CAP_DSHOW first on Windows
    camera = cv.VideoCapture(cam_id)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    base_path = pathlib.Path().joinpath(disk_path)
    os.makedirs(base_path, exist_ok=True)

    cv.namedWindow(preview_name)

    if camera.isOpened():  # try to get the first frame
        rval, frame = camera.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")

        return

    # to start frame motion-detection and writing press SPACE, motion is detected by frame-diff with last_frame
    # to pause frame writing, press ENTER

    # folder structure:
    # "three_darts_{time_s}                     gets created by pressing SPACE
    # --> {cam_id}
    #     --> "motion_series_"{time_s_new}
    #         --> {time_ns}_base_frame.png
    #             {time_ns}_frame.png

    three_darts_folder_prefix = "three_darts_"
    three_darts_folder_path = None
    motion_series_folder_path = None
    dart_impacting = False
    last_frame = None

    pause_frame_writing = True
    pause_last_time_pressed = time.time() - 10  # in seconds. Minus, so space and enter keys have no downtime at start
    play_last_time_pressed = time.time() - 10

    while rval:
        rval, frame = camera.read()

        if last_frame is not None:

            if not pause_frame_writing:

                if not motion_is_detected_slow(last_frame, frame):
                    dart_impacting = False
                else:
                    time_now_nanoseconds_frame = time.time_ns()

                    if not dart_impacting:
                        # start of a motion series
                        time_now_seconds_folder = time.time()
                        motion_series_folder_name = "motion_series_" + str(time_now_seconds_folder)
                        motion_series_folder_path = os.path.join(
                            base_path,
                            three_darts_folder_path,
                            motion_series_folder_name
                        )
                        os.makedirs(motion_series_folder_path, exist_ok=True)
                        dart_impacting = True
                        base_frame = last_frame
                        path_frame = os.path.join(
                            motion_series_folder_path,
                            f"{time_now_nanoseconds_frame}_base_frame.png"
                        )
                        cv.imwrite(path_frame, base_frame)

                    # continued motion series
                    path_frame = os.path.join(
                        motion_series_folder_path,
                        f"{time_now_nanoseconds_frame}_frame.png")
                    cv.imwrite(path_frame, frame)

        cv.imshow(preview_name, frame)
        cv.waitKey(1)

        last_frame = frame

        if keyboard.is_pressed('space'):
            play_pressed_current_time = time.time()  # in seconds
            if play_pressed_current_time - play_last_time_pressed > 8:  # seconds
                time_now_seconds = int(time.time())  # time in seconds since epoch
                time_now_nanoseconds = time.time_ns()
                three_darts_folder_name = three_darts_folder_prefix + str(time_now_seconds)
                three_darts_folder_path = os.path.join(base_path, three_darts_folder_name, str(cam_id))
                os.makedirs(three_darts_folder_path, exist_ok=True)
                path_frame = os.path.join(three_darts_folder_path, f"{time_now_nanoseconds}_empty_board.png")
                cv.imwrite(path_frame, frame)
                pause_frame_writing = False
                print(f"Cam {cam_id}: Frame writing online. ")

            play_last_time_pressed = play_pressed_current_time

        if keyboard.is_pressed('enter'):
            pause_current_time_pressed = time.time()  # in seconds
            if pause_current_time_pressed - pause_last_time_pressed > 4:  # seconds
                pause_frame_writing = True
                print(f"Cam {cam_id}: Frame writing paused. This line is longer so i can see it from afar.")

            pause_last_time_pressed = pause_current_time_pressed

        if keyboard.is_pressed('esc'):  # ESC key to exit
            print(f"Exiting {preview_name} recording.")
            break

    # cleanup after while loop
    camera.release()
    cv.destroyWindow(preview_name)


def start_webcam_threads(webcam_ids):
    for cam_id in webcam_ids:
        thread = CamThread("Camera " + str(cam_id), cam_id, save_motion_series)
        thread.start()


# cam_ids = list_connected_webcam_ids()
# start_webcam_threads(cam_ids)
# start_webcam_threads([0, 1])
start_webcam_threads([0, 1, 2])
