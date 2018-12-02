import cv2
import logging
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# fileConfig("logging_config.ini")
logger = logging.getLogger()


def load_image(source_path, image_filename):
    logger.info("Loading image from %s...", source_path)
    image = cv2.imread(source_path +image_filename)
    return image

def gray_image(image):
    logger.info("Image converted to grayscale")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


""" Subplot generator for images """
def plot(figure, subplot, image, title):
    figure.subplot(subplot)

    figure.imshow(image)
    figure.xlabel(title)
    figure.xticks([])
    figure.yticks([])
    return True


""" Show our results """
def show_results(original_image, gray_image, canny_image, auto_canny_image):
    plt.figure("test", figsize=(30,30))
    plot(plt, 321, original_image, "Original image")
    plot(plt, 322, gray_image, "Canny image")
    plot(plt, 323, canny_image, "Y bound")
    plot(plt, 324, auto_canny_image, "X bound")

    # if plate_image is not None:
    #     plot(plt, 324, plate_image, "License plate")
    #     plot(plt, 325, plate_image_char, "Characters outlined")
    #     plt.subplot(326)
    #     plt.text(0, 0, plate_number, fontsize=30)
    #     plt.xticks([])
    #     plt.yticks([])

    plt.tight_layout()
    plt.show()

    return True


def plot_histograms(hist):
    plt.plot(hist)
    plt.show()


