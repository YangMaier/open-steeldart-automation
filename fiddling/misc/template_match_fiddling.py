import pathlib

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt, patches
import matplotlib

from third_party.InvariantTM import invariant_match_template

img_empty_board_path1 = pathlib.Path("A:/dart_motion_series/1724320550845081700_1/1724320550845081700_0_base_frame.png")
img_empty_board_path2 = pathlib.Path("A:/dart_motion_series/1724320550871626700_0/1724320550871626700_0_base_frame.png")
img_board_path_1 = pathlib.Path("A:/dart_motion_series/1724320550845081700_1/1724320551650575400_1_frame.png")
img_board_path_2 = pathlib.Path("A:/dart_motion_series/1724320550871626700_0/1724320551284536700_1_frame.png")
img_empty_board_1 = cv.imread(str(img_empty_board_path1))
img_empty_board_2 = cv.imread(str(img_empty_board_path2))
img_board_1 = cv.imread(str(img_board_path_1))
img_board_2 = cv.imread(str(img_board_path_2))

path_template_folder_red = pathlib.Path().absolute().joinpath("../../src/media/contours/red")

# save all images in folder
templates_red = []
for path_template_red in path_template_folder_red.iterdir():
    img_template_red = cv.imread(str(path_template_red))
    templates_red.append(img_template_red)

# Convert it to grayscale
img_gray = cv.cvtColor(img_empty_board_1, cv.COLOR_BGR2GRAY)

x = 0


img_bgr = img_empty_board_1
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
template_bgr = templates_red[0]
template_rgb = cv.cvtColor(template_bgr, cv.COLOR_BGR2RGB)
cropped_template_rgb = template_rgb
cropped_template_rgb = np.array(cropped_template_rgb)
cropped_template_gray = cv.cvtColor(cropped_template_rgb, cv.COLOR_RGB2GRAY)
height, width = cropped_template_gray.shape
fig = plt.figure(num='Template - Close the Window to Continue >>>')
plt.imshow(cropped_template_rgb)
plt.show()
points_list = invariant_match_template(
    rgbimage=img_rgb,
    rgbtemplate=cropped_template_rgb,
    method="TM_CCOEFF_NORMED",
    matched_thresh=0.8,
    rot_range=[0,360],
    rot_interval=10,
    scale_range=[100,150],
    scale_interval=10,
    rm_redundant=True,
    minmax=True
)
fig, ax = plt.subplots(1)
plt.gcf().canvas.manager.set_window_title('Template Matching Results: Rectangles')
ax.imshow(img_rgb)
centers_list = []
for point_info in points_list:
    point = point_info[0]
    angle = point_info[1]
    scale = point_info[2]
    print(f"matched point: {point}, angle: {angle}, scale: {scale}")
    centers_list.append([point, scale])
    plt.scatter(point[0] + (width/2)*scale/100, point[1] + (height/2)*scale/100, s=20, color="red_transparent")
    plt.scatter(point[0], point[1], s=20, color="green")
    rectangle = patches.Rectangle((point[0], point[1]), width*scale/100, height*scale/100, color="red_transparent", alpha=0.50, label='Matched box')
    box = patches.Rectangle((point[0], point[1]), width*scale/100, height*scale/100, color="green", alpha=0.50, label='Bounding box')
    transform = matplotlib.transforms.Affine2D().rotate_deg_around(point[0] + width/2*scale/100, point[1] + height/2*scale/100, angle) + ax.transData
    rectangle.set_transform(transform)
    ax.add_patch(rectangle)
    ax.add_patch(box)
    plt.legend(handles=[rectangle,box])
#plt.grid(True)
plt.show()
fig2, ax2 = plt.subplots(1)
plt.gcf().canvas.manager.set_window_title('Template Matching Results: Centers')
ax2.imshow(img_rgb)
for point_info in centers_list:
    point = point_info[0]
    scale = point_info[1]
    plt.scatter(point[0]+width/2*scale/100, point[1]+height/2*scale/100, s=20, color="red_transparent")
plt.show()