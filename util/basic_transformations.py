import cv2
import numpy as np


class BasicTransformations:

    def __init__(self, display_helper):
        self.display_helper = display_helper

    def gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def bilateral_filter(self, image):
        return cv2.bilateralFilter(image, 16, 24, 24)

    def histogram_equalization(self, image):
        return cv2.equalizeHist(image)

    def sobel_vertical_edge_detection(self, image):
        vertical_image = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        return self.normalize_sobel_to_cv8u(vertical_image)

    def sobel_horizontal_edge_detection(self, image):
        horizontal_image = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        return self.normalize_sobel_to_cv8u(horizontal_image)

    def normalize_sobel_to_cv8u(self, sobel_image):
        sobelx_64f = sobel_image - np.min(sobel_image)  # to have only positive values
        div = np.max(sobelx_64f) / 255  # calculate the normalize divisor
        sobel_8u = np.uint8(sobelx_64f / div)
        return sobel_8u

    def binary_threshold(self, image, thresh):
        _, threshed = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        return threshed

    def skeletonize(self, image):
        size = np.size(image)
        skeleton = np.zeros(image.shape, np.uint8)

        image = self.binary_threshold(image, 140)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(image, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(image, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            image = eroded.copy()

            zeros = size - cv2.countNonZero(image)
            if zeros == size:
                done = True

        return skeleton

    def morphological_opening(self, image, kernel_size=(3, 3), iterations=15):
        opening_mask = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel=opening_mask, iterations=iterations)
        return cv2.subtract(image, opening_image)

    def color_mask(self, image, color):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if color == 'yellow':
            lower_mask = np.array([10, 100, 100])  # Yellow
            upper_mask = np.array([60, 255, 255])  # Yellow
        elif color == 'green':
            lower_mask = np.array([73, 100, 100])  # Green
            upper_mask = np.array([93, 255, 255])  # Green
        elif color == 'red':
            lower_mask = np.array([0, 30, 60])  # Red
            upper_mask = np.array([10, 120, 100])  # Red
        elif color == 'blue':
            lower_mask = np.array([20, 100, 100])  # Blue
            upper_mask = np.array([30, 255, 255])  # Blue
        else:
            raise Exception('Specified color not supported')

        mask = cv2.inRange(image_hsv, lower_mask, upper_mask)
        return mask
