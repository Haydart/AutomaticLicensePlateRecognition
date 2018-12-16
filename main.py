import os

from band_clipping import BindsFinder
from datasets import DatasetsProvider, samples, sample
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
    bands = bf.find_bands()

    return bands, canny_image



def skeletonized_sobel_method(image):
    sobel_vertical_image = sobel_vertical_edge_detection(image)
    skeletonized_sobel_vertical_image = skeletonization(sobel_vertical_image)

    bf = BindsFinder(skeletonized_sobel_vertical_image)
    bands = bf.find_bands()
    return bands


def opening_method(image):
    histogram_equalized_image = histogram_equalization(image)
    subtracted_image = morphological_opening(histogram_equalized_image)
    threshed_image = binary_threshold(subtracted_image, 100)

    bf = BindsFinder(threshed_image)
    bands = bf.find_bands()

    return bands, threshed_image


def run_pipelines_real_dataset():
    dp = DatasetsProvider(
        source_path='/home/lukasz/Studia/Analiza obrazow i wideo/UFPR-ALPR dataset/'
    )
    for image, position, number in dp.images():
        grayscale_image = gray_scale(image)
        noise_removed_image = bilateral_filter(grayscale_image)

        sobel_bands = skeletonized_sobel_method(noise_removed_image)
        canny_bands = canny_method(noise_removed_image)
        thresh_bands = opening_method(noise_removed_image)

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

        canny_bands, img1 = canny_method(noise_removed_image)
        thresh_bands, img = opening_method(noise_removed_image)
        sobel_bands, img2 = skeletonized_sobel_method(noise_removed_image)

        for band in canny_bands:
            show_bounds(image, band, GREEN)

        for band in thresh_bands:
            show_bounds(image, band, RED)

        for band in sobel_bands:
            show_bounds(image, band, BLUE)

        save_image(image, number, '')
        print("image done")


def process():
    # run_pipelines_sample_dataset()

    image, name = sample('043')
    grayscale_image = gray_scale(image)
    noise_removed_image = bilateral_filter(grayscale_image)
    sobel = sobel_horizontal_edge_detection(noise_removed_image)
    skeletonized_sobel_vertical, sobel_thresh = skeletonization(sobel)

    bf = BindsFinder(skeletonized_sobel_vertical)
    bands = bf.find_bands()

    for band in bands:
        show_bounds(image, band, RED)

    plot_image(grayscale_image, 1, 'grayscale')
    plot_image(noise_removed_image, 2, 'bilateral')
    plot_image(sobel, 3, 'vertical sobel')
    plot_image(sobel_thresh, 4, 'vertical sobel adt threshold')
    plot_image(skeletonized_sobel_vertical, 5, 'vertical sobel skeleton')
    plot_image(image, 6, 'image with bounds')


    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    fig = plt.gcf()
    fig.set_size_inches(10, 15)
    print('calculated')
    plt.show()


if __name__ == '__main__':
    process()
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
