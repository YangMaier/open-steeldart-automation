import time

import cv2 as cv
import numpy as np

from skimage.filters import gaussian
from skimage import img_as_ubyte
import skimage as ski

from data_structures.segment_template import ExpectedTemplatesWhite, FoundSegments, ExpectedTemplatesBlack
from fiddling.sliders.kmeans_masks import get_relevant_segments


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
    cv.createTrackbar('thresh white', 'sliders', 0, 255, nothing)
    cv.createTrackbar('thresh black', 'sliders', 0, 255, nothing)

    cv.setTrackbarPos('thresh white', 'sliders', 50)  # 55
    cv.setTrackbarPos('thresh black', 'sliders', 123)  # 140

    while True:
        frame_time = time.time_ns()

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        ski_img = ski.util.img_as_float(rgb)

        gaussian_img = gaussian(ski_img, 1)

        cv_image = img_as_ubyte(gaussian_img)
        bgr = cv.cvtColor(cv_image, cv.COLOR_RGB2BGR)

        grey = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

        thresh_white = cv.getTrackbarPos('thresh white', 'sliders')
        thresh_black = cv.getTrackbarPos('thresh black', 'sliders')

        mask_white = cv.threshold(grey, thresh_white, 255, cv.THRESH_BINARY)[1]
        expected_white_segments = ExpectedTemplatesWhite()
        found_segments_white: FoundSegments = get_relevant_segments([mask_white], [expected_white_segments])
        segments_white_img = np.zeros(frame.shape, dtype='uint8')
        cv.drawContours(segments_white_img, found_segments_white.inner_contours, -1, (255, 0, 255), -1)
        cv.drawContours(segments_white_img, found_segments_white.outer_contours, -1, (0, 255, 255), -1)
        cv.imshow("white segments", segments_white_img)

        mask_black = cv.threshold(grey, thresh_black, 255, cv.THRESH_BINARY)[1]
        mask_inverted = cv.bitwise_not(mask_black)
        expected_black_segments = ExpectedTemplatesBlack()
        found_segments_black: FoundSegments = get_relevant_segments([mask_inverted], [expected_black_segments])
        segments_black_img = np.zeros(frame.shape, dtype='uint8')
        cv.drawContours(segments_black_img, found_segments_black.inner_contours, -1, (255, 0, 255), -1)
        cv.drawContours(segments_black_img, found_segments_black.outer_contours, -1, (0, 255, 255), -1)
        cv.imshow("black segments", segments_black_img)

        if len(found_segments_white.outer_contours) == 10:
            found_inner_and_outer_segments = found_segments_white
        elif len(found_segments_black.outer_contours) == 10:
            found_inner_and_outer_segments = found_segments_black
        else:
            print("No inner and outer segments could be extracted.")
            found_inner_and_outer_segments = None

        if found_inner_and_outer_segments is not None:
            found_inner_and_outer_segments: FoundSegments
            all_outer_contours = [cnt for cnt in found_inner_and_outer_segments.outer_contours]
            outer_contours_combined = np.vstack(all_outer_contours)
            convex_hull_outer_segments = cv.convexHull(outer_contours_combined)
            (x, y), (a, b), angle = cv.fitEllipse(convex_hull_outer_segments)
            a = a * 1.12
            b = b * 1.12
            ellipse_around_dartboard = (x, y), (a, b), angle
            ellipse_mask = np.zeros(frame.shape[:2], dtype='uint8')
            cv.ellipse(ellipse_mask, ellipse_around_dartboard, (255, 255, 255), -1)
            masked_frame = cv.bitwise_and(frame, frame, mask=ellipse_mask)
            cv.imshow("ellipse masked frame", masked_frame)

            


        # cv.imshow("sliders", frame)
        cv.imshow("mask white", mask_white)
        cv.imshow("mask black", mask_inverted)

        frame_time_end = time.time_ns()
        print(f"Frame time: {round((frame_time_end - frame_time) / 1000000, 2)} ms")

        rval, frame = cam.read()

        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            cv.destroyAllWindows()
            break


threshold_sliders()

# KMEANS layers backup code. Was unreliable and slow in fiddling.
# kmeans_layers, layered_img = get_kmeans_layers(masked_frame, 6)
# layers_grey = [cv.cvtColor(layer, cv.COLOR_BGR2GRAY) for layer in kmeans_layers]
# for i, layer in enumerate(kmeans_layers):
#     cv.imshow(f"kmeans layer {i}", layer)

# cv.imshow("layered image", layered_img)
# es_red = ExpectedTemplatesRed()
# es_green = ExpectedTemplatesGreen()
# es_white = ExpectedTemplatesWhite()
# es_black = ExpectedTemplatesBlack()
# found_segments_by_kmeans_layers = get_relevant_segments(
#     layers_grey,
#     [es_red, es_green, es_white, es_black]
# )
# masks = found_segments_by_kmeans_layers.get_segments_as_masks(frame.shape[1], frame.shape[0])
# cv.imshow("bulls_eye_mask", masks[0])
# cv.imshow("bull_mask", masks[1])
# cv.imshow("inner_mask", masks[2])
# cv.imshow("triple_mask", masks[3])
# cv.imshow("outer_mask", masks[4])
# cv.imshow("double_mask", masks[5])


# # SLIC segmentation backup code. Produced only a black image in fiddling and was way too slow anyway.
# rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
# ski_img = ski.util.img_as_float(rgb)
# labels = segmentation.slic(ski_img, compactness=30, n_segments=400, start_label=1)
# g = graph.rag_mean_color(ski_img, labels)
#
# labels2 = graph.merge_hierarchical(
#     labels,
#     g,
#     thresh=35,
#     rag_copy=False,
#     in_place_merge=True,
#     merge_func=merge_mean_color,
#     weight_func=_weight_mean_color,
# )
#
# out = color.label2rgb(labels2, ski_img, kind='avg', bg_label=0)
# out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))
# cv.imshow("segmented frame", out)
# cv.waitKey(0)


# A fast test did not yield any results because of a wrong size somewhere
# rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#             ski_img = ski.util.img_as_float(rgb)
#             flooded_img = ski.segmentation.flood(ski_img, (200, 300), tolerance=0.01)
#             cv.imshow("flooded", flooded_img)