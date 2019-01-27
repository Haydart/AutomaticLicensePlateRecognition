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

    connected_components(binarized_image)

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


def connected_components(binarized_image):
    eroded_image = bt.erosion(binarized_image)
    components_count, output, stats, centroids = cv2.connectedComponentsWithStats(eroded_image, connectivity=4)

    sizes = stats[:, -1]
    centroids_areas = np.column_stack((centroids, sizes))
    centroids_areas = np.delete(centroids_areas, 0, axis=0)
    sorted_centroids_areas = centroids_areas[centroids_areas[:, -1].argsort()[::-1]]  # sort descending by size column
    two_largest_sizes

    max_label = 1
    max_size = sizes[1]
    for i in range(2, components_count):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    # label_hue = np.uint8(179 * output / np.max(output))
    # blank_ch = 255 * np.ones_like(label_hue)
    # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # # set bg label to black
    # labeled_img[label_hue == 0] = 0
    display_helper.add_to_plot(img2, title="Connected components")


if __name__ == '__main__':
    import os

    dir_path = '../dataset/skewed_trimmed_samples/'
    for filename in os.listdir(dir_path):
        if filename.startswith("I0000"):
            process('{}{}'.format(dir_path, filename))
