import utils as util
import cv2
import imutils
import numpy as np
from band_clipping import BindsFinder

source_path = '/home/lukasz/Studia/Analiza obrazow i wideo/ANPR/dataset/'
file_name = 'test_065.jpg'

image = util.load_image(source_path, file_name)
image_grey = util.gray_image(image)

image_gaussian = cv2.GaussianBlur(image_grey, (5, 5), 0)
image_median = cv2.medianBlur(image_grey, 5)

image_auto_canny = imutils.auto_canny(image_gaussian)

image_canny = cv2.Canny(image_gaussian, 170, 200, apertureSize=3)


bf = BindsFinder(image_canny)
bands = bf.get_bands()
bands_new = bf.last_step(bands)

print(np.asarray(bands))
print(np.asarray(bands_new))

for y0,y1, x0, x1 in bands_new:
    util.show_results(image_gaussian, image_canny, image_canny[y0:y1, ...], image_canny[y0:y1, x0:x1])
