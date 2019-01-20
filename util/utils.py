import logging

import cv2
from matplotlib import pyplot as plt

logger = logging.getLogger()


def plot(figure, subplot, image, title):
    figure.subplot(subplot)

    figure.imshow(image)
    figure.xlabel(title)
    figure.xticks([])
    figure.yticks([])
    return True


def plot_(figure, subplot, image, title):
    figure.subplot(subplot)

    figure.plot(image)
    figure.xlabel(title)

    return True


def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bilateral_filter(image):
    return cv2.bilateralFilter(image, 16, 32, 32)


def show_results(original_image, gray_image, canny_image, auto_canny_image=None):
    plt.figure("test", figsize=(30, 30))
    plot(plt, 321, original_image, "Original image")
    plot(plt, 322, gray_image, "Canny image")
    plot(plt, 323, canny_image, "Y bound")
    # plot(plt, 324, auto_canny_image, "X bound")

    plt.tight_layout()
    plt.show()

    return True


def show_one_image(image):
    plt.figure("Show image", figsize=(30, 30))
    plot(plt, 111, image, "Original image")
    plt.tight_layout()
    plt.show()

    return True


def plot_histograms(hist_1, hist_2, title):
    plt.figure("Histograms", figsize=(10, 5))
    plot_(plt, 121, hist_1, "Before")
    plot_(plt, 122, hist_2, "After")

    plt.title(title)
    plt.show()
