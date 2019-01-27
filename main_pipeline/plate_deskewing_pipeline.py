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
    print('Processing {}...'.format(image_path))
    image = Image.open(image_path)
    contrast_image = ImageEnhance.Contrast(image)
    img = contrast_image.enhance(3)
    img = np.asarray(img)
    channels_list = cv2.split(img)
    contrast_image = cv2.merge([channels_list[2], channels_list[1], channels_list[0]])  # b, g, r

    display_helper.add_to_plot(contrast_image, title="Contrast bump")
    gray_image = bt.gray_scale(contrast_image)
    binarized_image = bt.binary_threshold(gray_image, 200)
    result_polygon = cf.find_plate_contours(binarized_image)
    print(result_polygon)

    eroded_image = bt.erosion(binarized_image)
    ret, labels = cv2.connectedComponents(eroded_image)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue == 0] = 0

    display_helper.add_to_plot(labeled_img, title="Connected components")

    polygon_image = cf.draw_plate_polygon(img, result_polygon)

    display_helper.add_to_plot(polygon_image, title="Approx polygon")

    display_helper.plot_results()
    display_helper.reset_subplot()
    print("DONE")


if __name__ == '__main__':
    import os

    dir_path = '../dataset/skewed_trimmed_samples/'
    for filename in os.listdir(dir_path):
        if filename.startswith("I0000"):
            process('{}{}'.format(dir_path, filename))
