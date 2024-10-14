# fix for very slow cv.VideoCapture startup time, has to be done before importing cv
# see https://github.com/opencv/opencv/issues/17687
import os

from skimage import color
from skimage.draw import ellipse_perimeter
from skimage.transform import hough_ellipse

from operations_on.hardware import list_connected_webcam_ids

# os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# needed for pypylon to detect cameras
# os.environ["PYLON_CAMEMU"] = "3"

# sadly the fixes above violate PEP 8: E402
import cv2 as cv
import threading
import numpy as np


# Test Setup:
# 3x Rollei R-Cam 100 (1080p, 30fps)
# Winmau Blade 4
# LED Ring, RGB

# Assumptions for a working project:
# Dartboard fields are near perfectly build and are not bent.
# The Cam can see the whole Dartboard.


class CamThread(threading.Thread):
    """
    A thread class for displaying a preview of a webcam.

    Attributes:
        previewName (str): The name of the preview window.
        camID (int): The ID of the webcam to preview.

    Methods:
        run(self)
            Displays a preview of the webcam associated with the thread.
    """
    def __init__(self, preview_name, cam_id):
        """
                Initializes a new instance of the CamThread class.

                Args:
                    preview_name (str): The name of the preview window.
                    cam_id (int): The ID of the webcam to preview.
                """
        threading.Thread.__init__(self)
        self.previewName = preview_name
        self.camID = cam_id

    def run(self):
        print("Starting " + self.previewName)
        cam_preview(self.previewName, self.camID)
        # cam_preview_parameters(self.previewName, self.camID)
        # cam_canny_edge_preview(self.previewName + "canny", self.camID)
        # ellipse_blob_detector_preview(self.previewName + "Ellipses", self.camID)
        # ellipse_preview_custom(self.previewName + "Ellipses Custom", self.camID)
        # ellipse_find_contours.ellipse_find_contours(self.previewName + "Ellipses Custom", self.camID)
        # cam_preview_parameter_hsv_sliders.cam_preview_parameters(self.previewName, self.camID)
        # ellipse_circle_transformation.get_transformation_points(self.previewName, self.camID)


def cam_preview(preview_name, cam_id):
    """
        Displays a preview of the specified webcam.

        Args:
            preview_name (str): The name of the window to display the preview in.
            cam_id (int): The ID of the webcam to preview.

        Returns:
            None

        Description:
            This function opens a window and displays a preview of the specified webcam.
            The `cam_id` parameter specifies the ID of the webcam to preview.
            The function will continue to display the preview window until the user closes it.

        Example:
            cam_preview("Webcam 1", 1)
        """
    cv.namedWindow(preview_name)
    camera = cv.VideoCapture(cam_id)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    if camera.isOpened():  # try to get the first frame
        rval, frame = camera.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    while rval:
        cv.imshow(preview_name, frame)
        rval, frame = camera.read()
        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            break
    camera.release()
    cv.destroyWindow(preview_name)


def cam_preview_parameters(preview_name, cam_id):
    """
        Displays a preview of the specified webcam.

        Args:
            preview_name (str): The name of the window to display the preview in.
            cam_id (int): The ID of the webcam to preview. Defaults to 0.

        Returns:
            None

        Description:
            This function opens a window and displays a preview of the specified webcam.
            The `cam_id` parameter specifies the ID of the webcam to preview. If `cam_id` is not specified,
            the function defaults to previewing the first webcam. The function will continue to display the preview
            window until the user closes it.

        Example:
            cam_preview(1)
        """
    cam = cv.VideoCapture(cam_id)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    while rval:
        # frame = cv.bitwise_not(frame)  # invert color
        alpha = 1  # contrast
        beta = -60  # brightness
        frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
        cv.imshow("Frame", frame)

        # Set minimum and max HSV values to display
        lower_green = np.array([47, 151, 66])
        upper_green = np.array([96, 255, 255])
        # Create HSV Image and threshold into a range.
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask_green = cv.inRange(hsv, lower_green, upper_green)

        cv.imshow('Mask Green', mask_green)

        # Set minimum and max HSV values to display
        lower_red = np.array([0, 54, 82])
        upper_red = np.array([10, 255, 255])
        # Create HSV Image and threshold into a range.
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask_red = cv.inRange(hsv, lower_red, upper_red)

        cv.imshow('Mask Red', mask_red)

        masked_red_and_green = cv.bitwise_or(mask_green, mask_red)
        cv.imshow("Masked Red and Green", masked_red_and_green)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (3, 3), 0)
        cv.imshow("Gray and Blur", blur)

        # import high_contrast_image
        # high_contrast = high_contrast_image.high_contrast_image(blur)
        # cv.imshow("High Contrast", high_contrast)

        # brightness_mask = mask.brightness_mask(frame, 225)
        # masked_image = cv.bitwise_and(blur, blur, mask=brightness_mask)
        # cv.imshow("Mask Brightness", brightness_mask)

        # # filter out stuff that is too dark
        # threshold = 100
        # assignvalue = 255  # Value to assign the pixel if the threshold is met
        # _, threshold_black_img = cv.threshold(blur, threshold, assignvalue, cv.THRESH_BINARY)
        # cv.imshow("Threshold Dark", threshold_black_img)
        #
        # threshold_black_img = cv.bitwise_not(threshold_black_img)  # invert color
        #
        # # filter out stuff that is too bright
        # threshold = 200
        # assignvalue = 255  # Value to assign the pixel if the threshold is met
        # _, threshold_white_img = cv.threshold(threshold_black_img, threshold, assignvalue, cv.THRESH_BINARY)
        # cv.imshow("Threshold Bright", threshold_white_img)

        # threshold_white_img = cv.bitwise_not(frame)  # invert color

        # apply automatic Canny edge detection using the computed median
        median_with_fixed_offset = np.median(masked_red_and_green) + 40
        sigma = 0.33
        lower_t = int(max(0, (1.0 - sigma) * median_with_fixed_offset))
        upper_t = int(min(255, (1.0 + sigma) * median_with_fixed_offset))
        canny = cv.Canny(masked_red_and_green, lower_t, upper_t)
        cv.imshow("Auto Canny", canny)

        rval, frame = cam.read()
        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cam.release()
    cv.destroyAllWindows()



def cam_canny_edge_preview(preview_name, cam_id):
    # https://pyimagesearch.com/2021/05/12/opencv-edge-detection-cv-canny/

    # cv.namedWindow(preview_name)
    cam = cv.VideoCapture(cam_id)

    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    # blurs_params = [(3,3), (5, 5)]
    blurs_params = [(3, 3)]
    canny_params = [(10, 70), (30, 100), (50, 120), (80, 140), (100, 180), (120, 200), (140, 220), (150, 240), (180, 240), (190, 255), (210, 255), (220, 255), (230, 255), (240, 255), (250, 255)]
    # canny_params = [(50, 120)]
    for blur_param, canny_param in ((x, y) for x in blurs_params for y in canny_params):

        while rval:
            rval, frame = cam.read()
            key = cv.waitKey(20)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, blur_param, 0)
            canny = cv.Canny(blur, canny_param[0], canny_param[1], apertureSize=3)
            window_name = preview_name + ", blur: " + str(blur_param[0]) + " " + str(blur_param[1]) + ", canny: " + str(canny_param[0]) + " " + str(canny_param[1])
            cv.imshow(window_name, canny)
            if key == 32:  # next on SPACE
                cv.destroyWindow(window_name)
                break
            if key == 27:  # exit on ESC
                cv.destroyWindow(window_name)
                cam.release()
                return 0


def ellipse_blob_detector_preview(preview_name, cam_id):
    # https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/
    cv.namedWindow(preview_name)
    cam = cv.VideoCapture(cam_id)

    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    while rval:
        rval, frame = cam.read()
        key = cv.waitKey(20)
        # Apply canny to grayscale image. This ensures that there will be less noise during the edge detection process.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 1. Smoothing
        # 2. Computing image gradients
        # 3. Applying non - maxima suppression
        # 4. Utilizing hysteresis thresholding
        # Step 1 Smoothing: Smoothing an image allows us to ignore much of the detail and instead focus on the actual
        # structure. This also makes sense in the context of edge detection — we are not interested in the actual
        # detail of the image.
        # Instead, we want to apply edge detection to find the structure and outline of the objects in the image,
        # so we can further process them.
        # (7,7) yielded the least amount of noise in canny in a direct comparison of blur filters with
        # (3,3), (3,5), (5,5) and (7,7)
        blur = cv.GaussianBlur(gray, (3, 3), 0)
        # unclear: does another picture resolution result in another ksize optimum?

        # canny = cv.Canny(blur, 10, 70)
        # ret, mask = cv.threshold(canny, 120, 200, cv.THRESH_BINARY)
        # display the edge map
        # cv.imshow(preview_name, mask)

        # compute a "wide", "mid-range", and "tight" threshold for the edges
        # using the Canny edge detector
        # wide = cv.Canny(blur, 100, 140)
        # arguments: image, lower threshold, upper threshold
        # (100, 150) yielded best results with my setup
        canny = cv.Canny(blur, 50, 120)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        canny_thicker = cv.dilate(canny, kernel)

        # Set our filtering parameters
        # Initialize parameter setting using cv.SimpleBlobDetector
        params = cv.SimpleBlobDetector.Params()

        # Set Area filtering parameters
        params.filterByArea = True
        params.minArea = 5000

        # Set Circularity filtering parameters
        params.filterByCircularity = False
        params.minCircularity = 0.8

        # Set Convexity filtering parameters
        params.filterByConvexity = False
        params.minConvexity = 0.05

        # Set inertia filtering parameters
        params.filterByInertia = False
        params.minInertiaRatio = 9

        # Create a detector with the parameters
        detector = cv.SimpleBlobDetector.create(params)

        # Detect blobs
        keypoints = detector.detect(canny_thicker)

        # Draw blobs on our image as red_transparent circles
        blank = np.zeros((1, 1))
        blobs = cv.drawKeypoints(canny_thicker, keypoints, blank, (0, 0, 255),
                                  cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        number_of_blobs = len(keypoints)
        text = "Number of Circular Blobs: " + str(number_of_blobs)
        cv.putText(blobs, text, (20, 550),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

        # Show blobs
        cv.imshow(preview_name, blobs)

        if key == 27:  # exit on ESC
            break
    cam.release()
    cv.destroyWindow(preview_name)


def ellipse_preview_custom2(preview_name, cam_id):
    # https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/

    cv.namedWindow(preview_name)
    cam = cv.VideoCapture(cam_id)

    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    while rval:
        rval, frame = cam.read()
        key = cv.waitKey(20)
        # Apply canny to grayscale image. This ensures that there will be less noise during the edge detection process.
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 1. Smoothing
        # 2. Computing image gradients
        # 3. Applying non - maxima suppression
        # 4. Utilizing hysteresis thresholding
        # Step 1 Smoothing: Smoothing an image allows us to ignore much of the detail and instead focus on the actual
        # structure. This also makes sense in the context of edge detection — we are not interested in the actual
        # detail of the image.
        # Instead, we want to apply edge detection to find the structure and outline of the objects in the image,
        # so we can further process them.
        # (7,7) yielded the least amount of noise in canny in a direct comparison of blur filters with
        # (3,3), (3,5), (5,5) and (7,7)
        blur = cv.GaussianBlur(gray, (3, 3), 0)
        # unclear: does another picture resolution result in another ksize optimum?

        # canny = cv.Canny(blur, 10, 70)
        # ret, mask = cv.threshold(canny, 120, 200, cv.THRESH_BINARY)
        # display the edge map
        # cv.imshow(preview_name, mask)

        # compute a "wide", "mid-range", and "tight" threshold for the edges
        # using the Canny edge detector
        # wide = cv.Canny(blur, 100, 140)
        # arguments: image, lower threshold, upper threshold
        # (100, 150) yielded best results with my setup
        canny = cv.Canny(blur, 50, 120)

        result = hough_ellipse(canny, accuracy=20, threshold=250, min_size=100, max_size=1200)
        result.sort(order='accumulator')

        # Estimated parameters for the ellipse
        best = list(result[-1])
        yc, xc, a, b = (int(round(x)) for x in best[1:5])
        orientation = best[5]

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        # Draw the edge (white) and the resulting ellipse (red_transparent)
        edges = color.gray2rgb(img_as_ubyte(canny))
        edges[cy, cx] = (250, 0, 0)

        cv.imshow(preview_name, edges)

        # contours = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
        # contours_draw = cv.drawContours(canny, contours[200:250], -1, (255, 255, 0), 5)
        # Show blobs
        # print(len(contours))
        # cv.imshow(preview_name, contours_draw)

        if key == 27:  # exit on ESC
            break
    cam.release()
    cv.destroyWindow(preview_name)


def list_connected_webcams():
    """
        Returns a list of all connected webcams and their IDs.

        Returns:
            list: A list of tuples containing the webcam ID and serial number of each connected webcam.

        Description:
            This function uses the `pypylon` library to enumerate all available cameras and retrieve their serial numbers, effectively giving you a list of connected webcams and their IDs. The function returns a list of tuples, where each tuple contains the webcam ID and serial number of a connected webcam. The webcam ID is an integer representing the order in which the cameras were connected, starting from 0. The serial number is a unique identifier for each webcam.

        Example:
            list_connected_webcams()
            # Returns: [0, 1]
        """
    # Create an empty list to store camera IDs
    camera_ids = []

    # Get all the available cameras
    pylon_tl_factory = pylon.TlFactory.GetInstance()
    devices = pylon_tl_factory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    # Print the ID of each connected camera
    for i, cam in enumerate(devices):
        camera_ids.append(i)
        print(f"Webcam ID: {i}, Serial Number: {cam.GetSerialNumber()}")

    return camera_ids


def cam_test():
    cams_test = 500
    for i in range(0, cams_test):
        try:
            cap = cv.VideoCapture(i)
            test, frame = cap.read()
            if test:
                print("i : " + str(i) + " /// result: " + str(test))
        except:
            continue


# cam_test()


def open_webcams(webcam_ids):
    for cam_id in webcam_ids:
        thread = CamThread("Camera " + str(cam_id), cam_id)
        thread.start()


# open_webcams(cam_ids)
# open_webcams([0])
# open_webcams([1])
# open_webcams([1, 2, 3, 4, 5, 6])

cam_ids = list_connected_webcam_ids()
# open_webcams(cam_ids)
