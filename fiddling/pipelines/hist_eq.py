import time

import cv2 as cv
import numpy as np
from skimage.color import label2rgb

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
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Cam " + str(cam_id) + " could not be opened.")
        return

    while True:
        frame_time = time.time_ns()

        blur = cv.blur(frame, (5, 5))

        ycrcb_img = cv.cvtColor(blur, cv.COLOR_BGR2YCrCb)
        y, cr, cb = cv.split(ycrcb_img)
        y_equalized = cv.equalizeHist(y)
        ycrcb = cv.merge((y_equalized, cr, cb))
        equalized_ycrcb = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

        hsv_img = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_img)
        h_equalized = cv.equalizeHist(h)
        hsv = cv.merge((h_equalized, s, v))
        equalized_hsv = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        # corrected_img_path = pathlib.Path().absolute().joinpath("../../src/media/fiddling/empty_board_1080p_distance_mid_colors_corrected.png")
        # reference_img = cv.imread(str(corrected_img_path))
        # reference_img = cv.cvtColor(reference_img, cv.COLOR_BGR2RGB)
        # rgb = cv.cvtColor(equalized_ycrcb, cv.COLOR_BGR2RGB)
        # from skimage.exposure import match_histograms
        # matched = match_histograms(rgb, reference_img, channel_axis=-1)
        # matched_bgr = cv.cvtColor(matched, cv.COLOR_RGB2BGR)
        # mean_seed = np.average(h[345:355, 245:255])
        # flooded = ski.segmentation.flood_fill(h, (350, 250), tolerance=40, new_value=mean_seed)

        h_igg = inverse_gaussian_gradient(h_equalized)
        h_igg_cv = label2rgb(h_igg, h_equalized, kind='avg')

        # cv.imshow("flooded", flooded)
        cv.imshow("equalized hsv", equalized_hsv)
        cv.imshow("frame", frame)

        frame_time_end = time.time_ns()
        print(f"Frame time: {round((frame_time_end - frame_time) / 1000000, 2)} ms")

        rval, frame = cam.read()

        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            cv.destroyAllWindows()
            break


threshold_sliders()