import time

import cv2 as cv
import numpy as np

from skimage import img_as_ubyte

from skimage.segmentation import inverse_gaussian_gradient


def nothing(x):
    pass


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (
            graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count']
    )


def threshold_sliders():
    cam_id = 2
    cam = cv.VideoCapture(cam_id)
    # cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    # cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
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

    cv.namedWindow('sliders')
    cv.createTrackbar('alpha', 'sliders', 0, 1000, nothing)
    cv.createTrackbar('sigma', 'sliders', 0, 100, nothing)

    cv.setTrackbarPos('alpha', 'sliders', 400)
    cv.setTrackbarPos('sigma', 'sliders', 50)

    while True:
        frame_time = time.time_ns()

        blur = cv.blur(frame, (5, 5))

        ycrcb_img = cv.cvtColor(blur, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(ycrcb_img)
        y_equalized = cv.equalizeHist(y)
        ycrcb = cv.merge((y_equalized, cr, cb))
        equalized_bgr = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

        alpha_slider = cv.getTrackbarPos('alpha', 'sliders')
        alpha_slider = alpha_slider / 10
        sigma_slider = cv.getTrackbarPos('sigma', 'sliders')

        # ski_img = ski.util.img_as_float(rgb)

        # gaussian_img = gaussian(ski_img, 1)
        # quick_s = ski.segmentation.quickshift(gaussian_img)
        # out = color.label2rgb(quick_s, ski_img, kind='avg')
        # cv_image = img_as_ubyte(out)
        # bgr = cv.cvtColor(cv_image, cv.COLOR_RGB2BGR)
        # cv.imshow("inverse gaussian gradient", bgr)

        h, s, v = cv.split(cv.cvtColor(equalized_bgr, cv.COLOR_BGR2HSV))
        sigma_slider = sigma_slider / 100

        mask_red_1 = np.ma.masked_where(h < 10, h).mask
        mask_red_2 = np.ma.masked_where(h > 150, h).mask
        mask_red = np.ma.mask_or(mask_red_1, mask_red_2).mask
        # hist = cv.calcHist([h], [0], None, [256], [0, 256])
        # # normalize the histogram
        # hist /= hist.sum()
        # # plot the normalized histogram
        # plt.figure()
        # plt.title("Grayscale Histogram (Normalized)")
        # plt.xlabel("Bins")
        # plt.ylabel("% of Pixels")
        # plt.plot(hist)
        # plt.xlim([0, 256])
        # plt.show()


        igg_img = inverse_gaussian_gradient(h, alpha=alpha_slider, sigma=sigma_slider)
        cv_igg_img = img_as_ubyte(igg_img)
        # mgac_img = ski.segmentation.morphological_geodesic_active_contour(igg_img, num_iter=10, smoothing=2)
        # cv_mgac_img = label2rgb(mgac_img, equalized_image, kind='avg')
        # cv.imshow("inverse gaussian gradient", cv_image)

        # mgac = ski.segmentation.morphological_geodesic_active_contour(
        #     inverse_gaussian_gradient_img,
        #     num_iter=10,
        #     smoothing=2
        # )
        # cv.imshow("mgac", mgac)

        cv.imshow("equalized", equalized_image)
        cv.imshow("sliders", cv_igg_img)

        frame_time_end = time.time_ns()
        print(f"Frame time: {round((frame_time_end - frame_time) / 1000000, 2)} ms")

        rval, frame = cam.read()

        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            cv.destroyAllWindows()
            break


threshold_sliders()