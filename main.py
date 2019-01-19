import os

from final_solution.src.band_clipping import BindsFinder
from datasets import DatasetsProvider, samples, sample
from utils import *
import cv2
import numpy as np

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

    return bands


def run_pipelines_real_dataset():
    dp = DatasetsProvider(
        source_path='/home/lukasz/Studia/Analiza obrazow i wideo/Datasets/UFPR-ALPR dataset/'
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

    image, name = sample('006')
    grayscale_image = gray_scale(image)
    noise_removed_image = bilateral_filter(grayscale_image)

    horizontal_sobel = sobel_horizontal_edge_detection(noise_removed_image)
    cv2.threshold(horizontal_sobel, 135, 255, cv2.THRESH_TOZERO, horizontal_sobel)
    horizontal_sobel_skeleton, horizontal_sobel_thresh = skeletonization(horizontal_sobel)

    vertical_sobel = sobel_vertical_edge_detection(noise_removed_image)
    vertical_sobel_skeleton, vertical_sobel_thresh = skeletonization(vertical_sobel)

    bf = BindsFinder(horizontal_sobel_skeleton)
    bands = bf.find_bands()

    for band in bands:
        show_bounds(image, band, RED)

    plot_image(grayscale_image, 1, 'grayscale')
    plot_image(noise_removed_image, 2, 'bilateral')
    plot_image(image, 3, 'image with bounds')
    plot_image(horizontal_sobel, 3, 'horizontal sobel')
    plot_image(horizontal_sobel_thresh, 4, 'horizontal sobel threshold')
    plot_image(horizontal_sobel_skeleton, 5, 'horizontal sobel skeleton')
    plot_image(vertical_sobel, 6, 'vertical sobel')
    plot_image(vertical_sobel_thresh, 7, 'vertical sobel threshold')
    plot_image(vertical_sobel_skeleton, 8, 'vertical sobel skeleton')

    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    fig = plt.gcf()
    fig.set_size_inches(10, 15)
    print('calculated')
    plt.show()


if __name__ == '__main__':
    process()
    # sample_dataset()
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

    path = '/home/lukasz/Studia/Analiza obrazow i wideo/ALPR/SimpleALPR/dataset/P1010002.jpg'
    # Read image
    # im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Standard imports
    # import cv2
    # import numpy as np
    #
    # path = '/home/lukasz/Studia/Analiza obrazow i wideo/ALPR/SimpleALPR/dataset/P1010003.jpg'
    # # Read image
    # # im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # #
    # # # Set up the detector with default parameters.
    # # detector = cv2.SimpleBlobDetector_create()
    # #
    # # # Detect blobs.
    # # keypoints = detector.detect(im)
    # #
    # # # Draw detected blobs as red circles.
    # # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
    # #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    # im = cv2.imread(path)
    # hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    #
    # lower_red = np.array([20, 100, 100]) # Yellow
    # upper_red = np.array([30, 255, 255]) # Yellow
    #
    # # lower_red = np.array([0, 100, 50])  # Green
    # # upper_red = np.array([100, 255, 255])  # Green
    #
    # mask = cv2.inRange(hsv, lower_red, upper_red)
    # res = cv2.bitwise_and(im, im, mask=mask)
    #
    #
    # plt.imshow(res)
    # plt.show()
    # # Show keypoints
    # # cv2.imshow("Keypoints", im_with_keypoints)
    # # cv2.waitKey(0)

    lower_red = np.array([0, 100, 50])  # Green
    upper_red = np.array([100, 255, 255])  # Green

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(im, im, mask=mask)

    plt.imshow(res)
    plt.show()
    # Show keypoints
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)
