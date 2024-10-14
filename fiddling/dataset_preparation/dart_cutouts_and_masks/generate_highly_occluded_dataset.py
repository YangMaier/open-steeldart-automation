import pathlib

import imutils
import numpy as np

import cv2 as cv

from data_structures.coordinates import Coordinate2d


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    """ Layers a foreground image with alpha channel on top of a background image without alpha channel

    Args:
        background: image without alpha channel
        foreground: image with alpha channel
        x_offset: optional x offset
        y_offset: optional y offset

    Returns:
        background with foreground layered on top, no alpha channel

    References:
        https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
    """
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:

        return background

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

    return background


def rotate_img(img, angle):
    # angle in degrees
    # rotate_bound expands the image
    rotated_img = imutils.rotate_bound(img, angle)

    return rotated_img


def scale_img(img, scale_x, scale_y):
    resized_img = cv.resize(img, (0, 0), fx=scale_x, fy=scale_y)

    return resized_img


def get_dart_variation_img(img_dart_cropped, img_width, img_height, top_left_corner_base: Coordinate2d, scale_factor=0):
    bc_variation = get_base_coordinate_variation(top_left_corner_base)

    # rotate cropped dart img by a random small angle
    random_rotation = np.random.randint(low=-20, high=20)
    test_dart_r = rotate_img(img_dart_cropped, random_rotation)
    test_dart_r_s = scale_img(test_dart_r, 1 + scale_factor, 1 + scale_factor)
    img_dart_height_r = test_dart_r_s.shape[0]
    img_dart_width_r = test_dart_r_s.shape[1]

    # expand cropped image to 1080x1920

    # copy 0,0 rx,ry to bcv - rx / 2, bcv -ry / 2
    img_dart_expanded = np.zeros((img_height, img_width, 4), np.uint8)
    img_dart_expanded[bc_variation.x:bc_variation.x + img_dart_height_r, bc_variation.y:bc_variation.y + img_dart_width_r] = test_dart_r_s

    # get biggest_contour points
    grey_img_dart_cutout = cv.cvtColor(img_dart_expanded, cv.COLOR_BGRA2GRAY)
    contours, _ = cv.findContours(grey_img_dart_cutout, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv.contourArea)

    return img_dart_expanded, biggest_contour


def get_base_coordinate_variation(top_left_corner_base):
    base_coordinate_variation_distance = 50
    base_coordinate_variation_x = np.random.randint(low=top_left_corner_base.x - base_coordinate_variation_distance,
                                                    high=top_left_corner_base.x + base_coordinate_variation_distance,
                                                    dtype=np.uint16)
    base_coordinate_variation_y = np.random.randint(low=top_left_corner_base.y - base_coordinate_variation_distance,
                                                    high=top_left_corner_base.y + base_coordinate_variation_distance,
                                                    dtype=np.uint16)
    base_coordinate_variation = Coordinate2d(base_coordinate_variation_x, base_coordinate_variation_y)

    return base_coordinate_variation


def load_background_and_dart_images():
    dartboards_path = pathlib.Path().absolute().joinpath("backgrounds")
    dartboards = []
    for dartboard_path in dartboards_path.iterdir():
        img_dartboard = cv.imread(str(dartboard_path))
        dartboards.append(img_dartboard)
    dart_cutouts_path = pathlib.Path().absolute().joinpath("cropped")
    dart_cutout_images = []
    for dart_cutout_path in dart_cutouts_path.iterdir():
        img_dart_cutout = cv.imread(str(dart_cutout_path), cv.IMREAD_UNCHANGED)
        dart_cutout_images.append(img_dart_cutout)

    return dartboards, dart_cutout_images


dartboard_backgrounds, dart_cutout_images = load_background_and_dart_images()
background_quantity = len(dartboard_backgrounds)


# I want:
# 1000 one dart images
# 2000 two dart images
# 4000 three dart images, mostly tightly stacked
# 1000x 4, 5, ..., 12 dart images, randomly distributed
possible_dart_quantities = list(range(1, 13))
image_quantity = [1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# image_quantity = [1000, 2000, 4000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
for d_q, i_q in list(zip(possible_dart_quantities, image_quantity)):
    # get a random background
    # scale the cropped dart image down if the base coordinate y is lower and scale up if it is higher
    # save occlusion order
    # save occlusion masks, use all previous generated masks bitwise_or
    # save occlusion percentage
    # check if the current dart mask has enough visibility, if it is completely invisible, do something about it


# get occlusion level count for each 5%


get_dart_variation_img()
test_img = add_transparent_image(test_board, img_dart_expanded, 0, 0)

# save translated dart_n_contours in json format with fitting labels in COCO format
# unrotated box needed, top left corner points needed, box width and height needed
# "id": "1", 2, 3, 4?, 5?, 6?
# "type": "polygon",
# "label": "steeldart",
# "points": from RETR_LIST, APPROX_SIMPLE