import cv2


def ellipse_find_contours(preview_name, cam_id):
    # https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/

    cv2.namedWindow(preview_name)
    cam = cv2.VideoCapture(cam_id)

    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    while rval:
        rval, frame = cam.read()
        key = cv2.waitKey(20)
        # Apply canny to grayscale image. This ensures that there will be less noise during the edge detection process.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # unclear: does another picture resolution result in another ksize optimum?

        # canny = cv2.Canny(blur, 10, 70)
        # ret, mask = cv2.threshold(canny, 120, 200, cv2.THRESH_BINARY)
        # display the edge map
        # cv2.imshow(preview_name, mask)

        # compute a "wide", "mid-range", and "tight" threshold for the edges
        # using the Canny edge detector
        # wide = cv2.Canny(blur, 100, 140)
        # arguments: image, lower threshold, upper threshold
        # (100, 150) yielded best results with my setup
        canny = cv2.Canny(blur, 50, 120)


        cv2.imshow(preview_name, canny)

        contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours_draw = cv2.drawContours(canny, contours[200:250], -1, (255, 255, 0), 5)
        # Show blobs
        # print(len(contours))
        # cv2.imshow(preview_name, contours_draw)

        if key == 27:  # exit on ESC
            break
    cam.release()
    cv2.destroyWindow(preview_name)