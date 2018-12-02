import imutils

from band_clipping import BindsFinder
from utils import *

image = load_image('license_plate_snapshots/test_065.jpg')
image_grey = gray_scale(image)

gaussian_image = cv2.GaussianBlur(image_grey, (5, 5), 0)
median_image = cv2.medianBlur(image_grey, 5)

auto_canny_image = imutils.auto_canny(gaussian_image)

canny_image = canny_edge_detection(gaussian_image)

bf = BindsFinder(canny_image)
bands = bf.get_bands()
bands_new = bf.last_step(bands)

print(np.asarray(bands))
print(np.asarray(bands_new))

for y0, y1, x0, x1 in bands_new:
    show_results(gaussian_image, canny_image, canny_image[y0:y1, ...], canny_image[y0:y1, x0:x1])
