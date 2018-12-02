import logging

import cv2
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger()


def load_image(path):
    logger.info("Loading image from %s...", path)
    image = cv2.imread(path)
    return image


def plot(figure, subplot, image, title):
    figure.subplot(subplot)

    figure.imshow(image)
    figure.xlabel(title)
    figure.xticks([])
    figure.yticks([])
    return True


mpl.rcParams['figure.dpi'] = 150
subplot_width = 3
subplot_height = 5


def plot_image(img, subplot_index, title='', fix_colors=True):
    plt.subplot(subplot_height, subplot_width, subplot_index)

    if fix_colors:
        if len(img.shape) == 3 and img.shape[2] == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

    plt.title(title)
    plt.axis('off')


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bilateral_filter(img):
    return cv2.bilateralFilter(img, 32, 40, 40)


def equalize_histogram(img):
    return cv2.equalizeHist(img)


def morphological_opening(img):
    opening_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=opening_mask, iterations=15)
    return cv2.subtract(img, opening_image)


def morphological_closing(img, iterations=7):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def erosion(img):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(img, kernel, iterations=1)


def canny_edge_detection(img):
    return cv2.Canny(img, 170, 200)


def show_results(original_image, gray_image, canny_image, auto_canny_image):
    plt.figure("test", figsize=(30, 30))
    plot(plt, 321, original_image, "Original image")
    plot(plt, 322, gray_image, "Canny image")
    plot(plt, 323, canny_image, "Y bound")
    plot(plt, 324, auto_canny_image, "X bound")

    plt.tight_layout()
    plt.show()

    return True


def plot_histograms(hist):
    plt.plot(hist)
    plt.show()
