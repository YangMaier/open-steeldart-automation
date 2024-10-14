import os
import pathlib
import time

import cv2 as cv

from operations_on.images import motion_is_detected_slow


def save_dart_impact_frame_series():

    cam_id = 2
    cam = cv.VideoCapture(cam_id, cv.CAP_DSHOW)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_path = pathlib.Path().joinpath("A:/dart_motion_series")
    # frame_path = pathlib.Path().absolute().joinpath("../media/")
    os.makedirs(frame_path, exist_ok=True)
    frame_basename = ''
    #base_path = os.path.join(frame_path, frame_basename)
    base_path = frame_path
    frame_num = 0

    digit = len(str(int(cam.get(cv.CAP_PROP_FRAME_COUNT))))

    # cv.namedWindow(preview_name)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    last_frame = None
    dart_impacting = False
    base_frame = None

    cam_frames = 0
    cam_stabilized = False

    current_path = None

    while rval:
        rval, frame = cam.read()

        if cam_frames < 10:
            cam_frames += 1
            continue  # cam needs some frames to stabilize

        if last_frame is not None:

            if motion_is_detected_slow(last_frame, frame):

                time_now = time.time_ns()

                if not dart_impacting:
                    # start of a motion series
                    current_path = os.path.join(base_path, f"{time_now}")
                    os.makedirs(current_path, exist_ok=True)
                    dart_impacting = True
                    base_frame = last_frame
                    path_frame = os.path.join(current_path, f"{time_now}_0_base_frame.png")
                    cv.imwrite(path_frame, base_frame)


                path_frame = os.path.join(current_path, f"{time_now}_1_frame.png")
                cv.imwrite(path_frame, frame)
                # mask = np.zeros(frame.shape[:2], dtype="uint8")
                # cv.drawContours(mask, [dart_contour], -1, 255, -1)
                # masked_frame = cv.bitwise_and(frame, frame, mask=mask)
                # cv.putText(
                #     masked_frame,
                #     f"Shape size: {cv.contourArea(dart_contour)}",
                #     (5, 30),
                #     cv.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (0, 255, 255),
                #     1,
                #     cv.LINE_AA
                # )

                # img_diff = get_img_diff_skikit_similarity(base_frame, frame)
                # path_diff = os.path.join(base_path, f"{time_now}_2_diff.png")
                # cv.imwrite(path_diff, img_diff)
                #
                # img_masked_diff = skikit_diff_dart_approx(base_frame, frame)
                # path_masked = os.path.join(base_path, f"{time_now}_3_masked.png")
                # cv.imwrite(path_masked, img_masked_diff)
                #
                # img_features = get_img_features(img_masked_diff)
                # path_features = os.path.join(base_path, f"{time_now}_4_features.png")
                # cv.imwrite(path_features, img_features)

            else:
                dart_impacting = False

        cv.imshow("frame", frame)
        last_frame = frame

        key = cv.waitKey(1)
        # if key == ord('c'):
        #     cv.imwrite('{}_{}.{}'.format(base_path, str(frame_num).zfill(digit), 'png'), frame)
        #     print("Screenshot saved!")
        #     frame_num += 1
        if key == 27:  # exit on ESC
            cam.release()
            cv.destroyAllWindows()
            break


save_dart_impact_frame_series()
