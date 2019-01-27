import cv2
import numpy as np
from PIL import Image, ImageEnhance

from util.basic_transformations import BasicTransformations
from util.image_display_helper import ImageDisplayHelper
from util.plate_connected_component import PlateConnectedComponentExtractor
from util.plate_contours import PlateContoursFinder

cf = PlateContoursFinder()
display_helper = ImageDisplayHelper(True, 2, 14)
bt = BasicTransformations(display_helper)
ex = PlateConnectedComponentExtractor(bt)


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

    plate_component_image = ex.extract_plate_connected_component(binarized_image)
    display_helper.add_to_plot(plate_component_image, title="Plate connected component")

    result_polygon = cf.find_plate_contours(plate_component_image)
    polygon_image = cf.draw_plate_polygon(img, result_polygon)
    display_helper.add_to_plot(polygon_image, title="Approx polygon")

    hough_lines(gray_image, img)

    display_helper.plot_results()
    display_helper.reset_subplot()
    print("DONE")


def hough_lines(gray_image, img):
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    min_line_length = 100
    max_line_gap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, min_line_length, max_line_gap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    display_helper.add_to_plot(img, title="Hough Lines Prob")


if __name__ == '__main__':
    import os

    dir_path = '../dataset/skewed_trimmed_samples/'
    for filename in os.listdir(dir_path):
        if filename.startswith("skewed_"):
            process('{}{}'.format(dir_path, filename))

# def deskewe(file_name):
