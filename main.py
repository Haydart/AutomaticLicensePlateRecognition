import imutils
from datasets import DatasetsProvider, samples, sample, samples_v2
from band_clipping import BindsFinder
from utils import *
import os


GREEN = (0, 255, 0)
RED = (255, 0, 0)


def show_bounds(img, band, color):
    x = band[2]
    y = band[0]
    x1 = band[3]
    y1 = band[1]

    cv2.rectangle(img, (x, y), (x1, y1), color, 2)


def save_image(image, number, position):
    save_path = '/home/lukasz/Studia/Analiza obrazow i wideo/ALPR/SimpleALPR/results'
    position = position.replace(' ', '')
    ext = '.png'
    file_name = '{}_{}_{}'.format(number, position, ext)
    file_path = os.path.join(save_path, file_name)
    print(file_path)
    cv2.imwrite(file_path, image)


def canny_method(image):
    canny_image = canny_edge_detection(image)

    bf = BindsFinder(canny_image)
    bands = bf.find_bands()

    return bands, canny_image

def sobel_method(image):
    canny_image = canny_edge_detection(image)

    bf = BindsFinder(canny_image)
    bands = bf.find_bands()
    return bands


def thresh_method(image):
    histogram_equalized_image = histogram_equalization(image)
    subtracted_image = morphological_opening(histogram_equalized_image)
    threshed_image = binary_threshold(subtracted_image, 100)

    bf = BindsFinder(threshed_image)
    bands = bf.find_bands()

    return bands, threshed_image


def real_dataset():
    dp = DatasetsProvider(
        source_path='/home/lukasz/Studia/Analiza obrazow i wideo/UFPR-ALPR dataset/'
    )
    for image, position, number in dp.images():
        grayscale_image = gray_scale(image)
        noise_removed_image = bilateral_filter(grayscale_image)

        canny_bands = canny_method(noise_removed_image)
        thresh_bands = thresh_method(noise_removed_image)

        for band in canny_bands:
            show_bounds(image, band, GREEN)

        for band in thresh_bands:
            show_bounds(image, band, RED)

        save_image(image, number, position)


def sample_dataset():
    for image, number in samples():
        grayscale_image = gray_scale(image)
        noise_removed_image = bilateral_filter(grayscale_image)

        canny_bands, img1 = canny_method(noise_removed_image)
        thresh_bands, img = thresh_method(noise_removed_image)

        for band in canny_bands:
            show_bounds(image, band, GREEN)

        for band in thresh_bands:
            show_bounds(image, band, RED)

        save_image(image, number, '')

if __name__ == '__main__':
    sample_dataset()
    # real_dataset()
    # image, name = sample('001')
    # grayscale_image = gray_scale(image)
    # noise_removed_image = bilateral_filter(grayscale_image)
    #
    # thresh_bands = thresh_method(noise_removed_image)
    #
    # for y0, y1, x0, x1 in thresh_bands:
    #     from matplotlib import pyplot as plt
    #     plt.imshow(image[y0:y1, x0:x1])
    #     plt.show()