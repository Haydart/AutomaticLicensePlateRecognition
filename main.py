import os

from band_clipping import BindsFinder
from datasets import DatasetsProvider, samples
from utils import *

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def show_bounds(img, band, color):
    x = band[2]
    y = band[0]
    x1 = band[3]
    y1 = band[1]

    cv2.rectangle(img, (x, y), (x1, y1), color, 2)


def save_image(image, number, position):
    save_path = 'output/dataset_pipelines/'
    position = position.replace(' ', '')
    ext = '.png'
    file_name = '{}_{}_{}'.format(number, position, ext)
    file_path = os.path.join(save_path, file_name)
    print(file_path)
    cv2.imwrite(file_path, image)


def canny_method(image):
    canny_image = canny_edge_detection(image)

    bf = BindsFinder(canny_image)
    bands = bf.get_bands()
    bands_new = bf.last_step(bands)
    return bands_new


def thresh_method(image):
    histogram_equalized_image = histogram_equalization(image)
    subtracted_image = morphological_opening(histogram_equalized_image)
    threshed_image = binary_threshold(subtracted_image, 100)

    bf = BindsFinder(threshed_image)
    bands = bf.get_bands()
    bands_new = bf.last_step(bands)
    return bands_new


def sobel_method(image):
    canny_image = canny_edge_detection(image)

    bf = BindsFinder(canny_image)
    bands = bf.get_bands()
    bands_new = bf.last_step(bands)
    return bands_new


def run_pipelines_real_dataset():
    dp = DatasetsProvider(
        source_path='/home/lukasz/Studia/Analiza obrazow i wideo/UFPR-ALPR dataset/'
    )
    for image, position, number in dp.images():
        grayscale_image = gray_scale(image)
        noise_removed_image = bilateral_filter(grayscale_image)
        sobel_bands = sobel_method(noise_removed_image)

        canny_bands = canny_method(noise_removed_image)
        thresh_bands = thresh_method(noise_removed_image)

        for band in canny_bands:
            show_bounds(image, band, GREEN)

        for band in thresh_bands:
            show_bounds(image, band, RED)

        for band in sobel_bands:
            show_bounds(image, band, BLUE)

        save_image(image, number, position)


def run_pipelines_sample_dataset():
    for image, number in samples():
        grayscale_image = gray_scale(image)
        noise_removed_image = bilateral_filter(grayscale_image)

        canny_bands = canny_method(noise_removed_image)
        thresh_bands = thresh_method(noise_removed_image)
        sobel_bands = sobel_method(noise_removed_image)

        for band in canny_bands:
            show_bounds(image, band, GREEN)

        for band in thresh_bands:
            show_bounds(image, band, RED)

        for band in sobel_bands:
            show_bounds(image, band, BLUE)

        save_image(image, number, '')


def process():
    run_pipelines_sample_dataset()


# image, name = sample('006')
# grayscale_image = gray_scale(image)
# noise_removed_image = bilateral_filter(grayscale_image)
# sobel = sobel_vertical_edge_detection(noise_removed_image)
# sobel_thresh = cv2.adaptiveThreshold(sobel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
# skeletonized_sobel_vertical = skeletonization(sobel)
#
# finder = BindsFinder(skeletonized_sobel_vertical)
# finder._find_y_bands()
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


if __name__ == '__main__':
    process()
