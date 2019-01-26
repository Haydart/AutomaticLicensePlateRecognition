import numpy as np
from PIL import Image, ImageEnhance
from cv2 import cv2

from util.basic_transformations import BasicTransformations
from util.image_display_helper import ImageDisplayHelper
from util.plate_contours import PlateContoursFinder

cf = PlateContoursFinder()
display_helper = ImageDisplayHelper(True, 2, 14)
bt = BasicTransformations(display_helper)


def process(image_path):
    image = Image.open(image_path)
    contrast_image = ImageEnhance.Contrast(image)
    img = contrast_image.enhance(3)
    img = np.asarray(img)
    channels_list = cv2.split(img)
    contrast_image = cv2.merge([channels_list[2], channels_list[1], channels_list[0]])  # b, g, r

    display_helper.add_to_plot(contrast_image, title="Contrast bump")
    gray_image = bt.gray_scale(contrast_image)
    binarized_image = bt.binary_threshold(gray_image, 210)
    result_polygon = cf.find_plate_contours(binarized_image)

    print(result_polygon)
    polygon_image = cf.draw_plate_polygon(img, result_polygon)
    display_helper.add_to_plot(polygon_image, title="Approx polygon")

    display_helper.plot_results()
    print("DONE")


if __name__ == '__main__':
    process('../dataset/skewed_trimmed_samples/I00006.png')
