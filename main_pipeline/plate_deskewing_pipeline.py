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
    centroids_areas = np.column_stack((
        np.arange(components_count, dtype=int),
        centroids,
        np.zeros(components_count),
        sizes
    ))

    centroids_areas_no_black = np.delete(centroids_areas, 0, axis=0)

    # sort descending by size column
    sorted_components_info = centroids_areas_no_black[centroids_areas_no_black[:, -1].argsort()[::-1]]

    largest_component_info = choose_plate_component(binarized_image, sorted_components_info)
    largest_component_label = largest_component_info[0]
    largest_components_image = np.zeros(output.shape)
    largest_components_image[output == largest_component_label] = 255
    display_helper.add_to_plot(largest_components_image, title="Connected components")


def choose_plate_component(image, sorted_components_info):
    largest_components_info = sorted_components_info[:2, :]
    image_center = image.shape[::-1]

    def calculate_centroid_distance(row):
        import math
        x_dist = image_center[0] / 2 - row[1]
        y_dist = image_center[1] / 2 - row[2]
        row[3] = math.sqrt(pow(x_dist, 2) + pow(y_dist, 2))
        return row

    # take largest component if it's much bigger than next largest
    # otherwise, choose component, for which distance from image center to its centroid is the smallest
    largest_component_area_trust_threshold = 4
    largest_components_sizes = np.sort(largest_components_info[:, -1])
    if largest_components_sizes[1] / largest_components_sizes[0] <= largest_component_area_trust_threshold:
        np.apply_along_axis(calculate_centroid_distance, 1, largest_components_info)
        # sort ascending by centroid-image center distance
        return largest_components_info[largest_components_info[:, -2].argsort()][0, :]
    else:
        return largest_components_info[0, :]


if __name__ == '__main__':
    import os

    dir_path = '../dataset/skewed_trimmed_samples/'
    for filename in os.listdir(dir_path):
        if filename.startswith("I0001"):
            process('{}{}'.format(dir_path, filename))
