
import cv2 as cv
import numpy as np

def nothing(x):
    pass


def threshold_sliders():
    cam_id = 2
    cam = cv.VideoCapture(cam_id)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    # base_frame_path = pathlib.Path().absolute().joinpath("../../fiddling/media/fiddling/1723902584967225400.png")
    # frame_path = pathlib.Path().absolute().joinpath("../../fiddling/media/fiddling/1724076018156267600_2_diff.png")
    # base_frame = cv.imread(str(base_frame_path))
    # frame = cv.imread(str(frame_path))
    # base_frame = cv.cvtColor(base_frame, cv.COLOR_BGR2GRAY)
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # os.makedirs(frame_path, exist_ok=True)
    # frame_basename = 'sample_video_cap'
    # base_path = os.path.join(frame_path, frame_basename)
    # frame_num = 0

    # digit = len(str(int(cam.get(cv.CAP_PROP_FRAME_COUNT))))

    # cv.namedWindow(preview_name)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    cv.namedWindow('threshhold')
    # cv.createTrackbar('blur', 'threshhold', 0, 10, nothing)
    cv.createTrackbar('thresh', 'threshhold', 0, 255, nothing)
    cv.createTrackbar('new_val', 'threshhold', 0, 255, nothing)
    cv.createTrackbar('gaussian_block_size', 'threshhold', 5, 500, nothing)
    cv.createTrackbar('gaussian_c', 'threshhold', 0, 100, nothing)

    cv.setTrackbarPos('thresh', 'threshhold', 82)
    cv.setTrackbarPos('new_val', 'threshhold', 255)
    cv.setTrackbarPos('gaussian_block_size', 'threshhold', 23)
    cv.setTrackbarPos('gaussian_c', 'threshhold', 8)

    while True:

        thresh = cv.getTrackbarPos('thresh', 'threshhold')
        new_val = cv.getTrackbarPos('new_val', 'threshhold')
        gaussian_block_size = cv.getTrackbarPos('gaussian_block_size', 'threshhold')
        if gaussian_block_size % 2 == 0:
            gaussian_block_size += 1
        if gaussian_block_size < 5:
            gaussian_block_size = 5
        gaussian_c = cv.getTrackbarPos('gaussian_c', 'threshhold')

        ycrcb_img = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(ycrcb_img)
        y_equalized = cv.equalizeHist(y)
        ycrcb = cv.merge((y_equalized, cr, cb))
        equalized_bgr = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)
        # cv.imshow("equalized bgr", equalized_bgr)

        grey = cv.cvtColor(equalized_bgr, cv.COLOR_BGR2GRAY)
        thresh_gaussian = cv.adaptiveThreshold(grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, gaussian_block_size, gaussian_c)
        cv.imshow("thresh gaussian", thresh_gaussian)
        # thresh_gaussian_dilated = cv.dilate(thresh_gaussian, np.ones((3, 3), np.uint8))
        # cv.imshow("thresh gaussian dilated", thresh_gaussian_dilated)
        # thresh_gaussian_inverted = cv.bitwise_not(thresh_gaussian)
        # cv.imshow("thresh gaussian inverted", thresh_gaussian_inverted)
        contours, _ = cv.findContours(thresh_gaussian, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours_filtered = [cnt for cnt in contours if 50 < cv.contourArea(cnt)]
        contours_img = np.zeros(frame.shape[:2], dtype="uint8")
        cv.drawContours(contours_img, contours_filtered, -1, (255, 255, 255), 1)
        # dilated_contours_img = cv.dilate(contours_img, np.ones((1, 1), np.uint8))
        cv.imshow("contours img thresh gaussian", contours_img)

        # hsv = cv.cvtColor(equalized_bgr, cv.COLOR_BGR2HSV)
        # h, s, v = cv.split(hsv)
        # s_v_sum = np.add(s, v, dtype=np.uint16)
        # s_v_sum = np.clip(s_v_sum, 0, 255).astype(np.uint8)
        # s_v_sum_inverted = cv.bitwise_not(s_v_sum)
        # _, thresh_s_v_sum_inv = cv.threshold(s_v_sum_inverted, 150, 255, cv.THRESH_BINARY)
        # cv.imshow("thresh s v sum inv", thresh_s_v_sum_inv)

        # # calculating object histogram
        # M = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # path_white_roi = pathlib.Path().absolute().joinpath("../../src/media/calibration/white_roi.png")
        # path_black_roi = pathlib.Path().absolute().joinpath("../../src/media/calibration/black_roi.png")
        # white_roi_img = cv.imread(str(path_white_roi))
        # black_roi_img = cv.imread(str(path_black_roi))
        # white_hsv_roi = cv.cvtColor(white_roi_img, cv.COLOR_BGR2HSV)
        # black_hsv_roi = cv.cvtColor(black_roi_img, cv.COLOR_BGR2HSV)

        # # Histogram ROI
        # roi_hist_white = cv.calcHist([white_hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # mask_white = cv.calcBackProject([hsv], [0, 1], roi_hist_white, [0, 180, 0, 256], 1)
        #
        # roi_hist_black = cv.calcHist([black_hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # mask_black = cv.calcBackProject([hsv], [0, 1], roi_hist_black, [0, 180, 0, 256], 1)
        #
        # ## Filtering remove noise
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)) # it will nothing but will create an array of given shape.
        # mask_white_filter = cv.filter2D(mask_white, -1, kernel)
        # _, mask_white_filter_thresh = cv.threshold(mask_white_filter, 50, 255, cv.THRESH_BINARY)
        # mask_white_end = cv.merge((mask_white_filter_thresh, mask_white_filter_thresh, mask_white_filter_thresh))
        # result_white = cv.bitwise_or(hsv, mask_white_end)
        #
        # ## Filtering remove noise
        # mask_black_filter = cv.filter2D(mask_black, -1, kernel)
        # _, mask_black_filter_thresh = cv.threshold(mask_black_filter, 50, 255, cv.THRESH_BINARY)
        # mask_black_end = cv.merge((mask_black_filter_thresh, mask_black_filter_thresh, mask_black_filter_thresh))
        # result_black = cv.bitwise_or(hsv, mask_black_end)

        # img_thresh_to_zero = cv.threshold(equalized_bgr, thresh, new_val, cv.THRESH_TOZERO)[1]
        # cv.imshow("thresh to zero", img_thresh_to_zero)
        #
        # img_thresh_binary = cv.threshold(equalized_bgr, thresh, 255, cv.THRESH_BINARY)[1]
        # cv.imshow("thresh binary", img_thresh_binary)

        diff_thresh_t = cv.threshold(equalized_bgr, thresh, new_val, cv.THRESH_TOZERO)[1]
        # contours, _ = cv.findContours(diff_thresh_t, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # contours_filtered = [cnt for cnt in contours if cv.contourArea(cnt) > 200]
        # mask_custom = np.zeros(frame.shape, dtype='uint8')
        # cv.drawContours(mask_custom, contours_filtered, -1, 255, -1)
        # dilate_mask = cv.dilate(mask_custom, np.ones((15, 15), np.uint8), iterations=1)
        # mask_applied = cv.bitwise_and(frame, frame, mask=dilate_mask)
        #
        # grey = cv.cvtColor(equalized_bgr, cv.COLOR_BGR2GRAY)
        # thresh_img = cv.threshold(grey, thresh, new_val, cv.THRESH_BINARY)[1]
        # thresh_inverted = cv.bitwise_not(thresh_img)
        # thresh_inverted_opening = cv.morphologyEx(thresh_inverted, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS,(5, 5)), iterations=2)
        # # thresh_img = cv.dilate(thresh_img, np.ones((3, 3), np.uint8), iterations=3)
        # # mask_inverted = cv.bitwise_not(mask)
        # contours, _ = cv.findContours(thresh_inverted_opening, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # contours_filtered = [cnt for cnt in contours if cv.contourArea(cnt) > 1000]
        # thresh_inverted_contours_filtered = np.zeros(frame.shape, dtype='uint8')
        # cv.drawContours(thresh_inverted_contours_filtered, contours_filtered, -1, (255, 255, 255), -1)
        #
        # cv.imshow("thresh inverted", thresh_inverted)
        # cv.imshow("thresh inverted opening", thresh_inverted_opening)
        # cv.imshow("thresh inverted contours filtered", thresh_inverted_contours_filtered)

        rval, frame = cam.read()

        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            cv.destroyAllWindows()
            break

    cam.release()

threshold_sliders()


