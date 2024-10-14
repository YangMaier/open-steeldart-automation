
import time

from calculate_score_board import get_board

# fix for very slow cv2.VideoCapture startup time
# os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv

vid = cv.VideoCapture(0)


def main():
    cam_id = 2
    cam = cv.VideoCapture(cam_id)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    while True:

        frame_time = time.time_ns()

        # get board here
        (
            calibration_completed,
            score_segments,
            equalized_bgr,
            masked_img_red,
            masked_img_green,
            img_not_matched_red,
            img_not_matched_green,
            img_bulls_eye_contours,
            img_bull_contours,
            img_triple_contours,
            img_double_contours,
            equalized_bgr_raw_radial_transitions,
            equalized_bgr_radial_sections,
            equalized_bgr_rings_smoothed,
            score_segment_img
        ) = get_board(frame)

        frame_time_end = time.time_ns()
        print(f"Frame time: {round((frame_time_end - frame_time) / 1000000, 2)} ms")

        cv.imshow("bulls eye", img_bulls_eye_contours)
        cv.imshow("bull", img_bull_contours)
        # cv.imshow("inner", img_inner_contours)
        cv.imshow("triple", img_triple_contours)
        # cv.imshow("outer", img_outer_contours)
        cv.imshow("double", img_double_contours)
        # cv.imshow("all contours", img_all_contours)

        cv.imshow("equalized", equalized_bgr)

        if calibration_completed:
            cv.imshow("score segment img", score_segment_img)

        rval, frame = cam.read()

        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            cv.destroyAllWindows()
            break

    cam.release()


if __name__ == "__main__":
    main()

#
#     arbitrary_impact_point_x = 200
#     arbitrary_impact_point_y = 300
#     arbitrary_impact_point = Coordinate2d(arbitrary_impact_point_x, arbitrary_impact_point_y)
#     draw_circle(img_saved, arbitrary_impact_point, (0, 255, 255), 3)
#     arb_score, arb_confidence = dart_board.get_score_and_confidence(arbitrary_impact_point)
#
#     dart_board_corner_points = get_board_points(dart_board)
#
#     transform_img(frame, dart_board_corner_points, board_corners_reference)
#
#     cv.imshow('frame', frame)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
#
# vid.release()
# cv.destroyAllWindows()
