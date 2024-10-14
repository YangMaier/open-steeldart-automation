import cv2
import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte, io
from skimage.feature import canny
from skimage.transform import hough_ellipse, rescale
from skimage.draw import ellipse_perimeter
import pathlib

# Load picture, convert to grayscale and detect edges
image_path = pathlib.Path().absolute().joinpath("../media/fiddling/cam_input.jpg")
# image_rgb = cv2.imread(str(image_path))
# image_rgb = data.coffee()[0:220, 160:420]
image_rgb = io.imread(str(image_path))
gray = color.rgb2gray(image_rgb)
rescaled = rescale(gray, 0.6, anti_aliasing=True)
# plt.imshow(rescaled)
# gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
print("doing edges")
edges = canny(rescaled, sigma=1.0, low_threshold=0.4, high_threshold=0.8)
plt.imshow(edges)
plt.show()

# Perform a Hough Transform
# The accuracy corresponds to the bin size of the histogram for minor axis lengths.
# A higher `accuracy` value will lead to more ellipses being found, at the
# cost of a lower precision on the minor axis length estimation.
# A higher `threshold` will lead to less ellipses being found, filtering out those
# with fewer edge points (as found above by the Canny detector) on their perimeter.
print("doing ellipse")
result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=120, max_size=450)
print("doing sorting")
result.sort(order='accumulator')

print("doing estimate")
# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = (int(round(x)) for x in best[1:5])
orientation = best[5]

print("doing draw")
# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
image_rgb[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red_transparent)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)

print("doing figures")
fig2, (ax1, ax2) = plt.subplots(
    ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True
)

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red_transparent)')
ax2.imshow(edges)

plt.show()