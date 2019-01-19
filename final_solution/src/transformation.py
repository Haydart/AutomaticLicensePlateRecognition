import cv2
import numpy as np
from copy import copy


class BasicTransforms:

    @staticmethod
    def gray_scale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def bilateral_filter(image):
        return cv2.bilateralFilter(image, 32, 40, 40)

    @staticmethod
    def histogram_equalization(image):
        return cv2.equalizeHist(image)

    @staticmethod
    def sobel_vertical_edge_detection(image):
        vertical_image = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        return BasicTransforms.normalize_sobel_to_cv8u(vertical_image)

    @staticmethod
    def sobel_horizontal_edge_detection(image):
        horizontal_image = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        return BasicTransforms.normalize_sobel_to_cv8u(horizontal_image)

    @staticmethod
    def normalize_sobel_to_cv8u(sobel_image):
        sobelx_64f = sobel_image - np.min(sobel_image)  # to have only positive values
        div = np.max(sobelx_64f) / 255  # calculate the normalize divisor
        sobel_8u = np.uint8(sobelx_64f / div)
        return sobel_8u

    @staticmethod
    def binary_threshold(image, thresh):
        _, threshed = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        return threshed

    @staticmethod
    def skeletonize(image):
        size = np.size(image)
        skeleton = np.zeros(image.shape, np.uint8)

        image = BasicTransforms.binary_threshold(image, 140)

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

    @staticmethod
    def morphological_opening(image, kernel_size=(3, 3), iterations=15):
        opening_mask = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel=opening_mask, iterations=iterations)
        return cv2.subtract(image, opening_image)

    @staticmethod
    def color_mask(image, color):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if color == 'yellow':
            lower_mask = np.array([10, 100, 100])  # Yellow
            upper_mask = np.array([60, 255, 255])  # Yellow
        elif color == 'green':
            lower_mask = np.array([73, 100, 100])  # Green
            upper_mask = np.array([93, 255, 255])  # Green
        elif color == 'red':
            lower_mask = np.array([0, 30, 60])     # Red
            upper_mask = np.array([10, 120, 100])  # Red
        elif color == 'blue':
            lower_mask = np.array([20, 100, 100])  # Blue
            upper_mask = np.array([30, 255, 255])  # Blue
        else:
            raise Exception('Specified color not supported')

        mask = cv2.inRange(image_hsv, lower_mask, upper_mask)
        return mask



class AdvancedTransforms:

    def __init__(self, transforms=BasicTransforms):
        self.transforms = transforms

    def preprocess(self, image):
        image = self.transforms.gray_scale(image)
        image = self.transforms.bilateral_filter(image)
        return image

    def skeletonized_sobel_method(self, image):
        image_vertical = self.transforms.sobel_vertical_edge_detection(copy(image))
        image_horizontal = self.transforms.sobel_horizontal_edge_detection(copy(image))

        image_vertical = self.transforms.skeletonize(image_vertical)
        image_horizontal = self.transforms.skeletonize(image_horizontal)

        return image_vertical, image_horizontal

    def opening_method(self, image):
        image = self.transforms.histogram_equalization(image)
        image = self.transforms.morphological_opening(image)
        image = self.transforms.binary_threshold(image, 100)
        return image

    def color_mask_method(self, image):
        image_yellow = self.transforms.color_mask(copy(image), 'yellow')
        image_red = self.transforms.color_mask(copy(image), 'red')
        image_green = self.transforms.color_mask(copy(image), 'green')
        # image = self.transforms.color_mask(copy(image), 'blue')

        # image_yellow = self.transforms.sobel_vertical_edge_detection(image_yellow)
        # image_red = self.transforms.sobel_vertical_edge_detection(image_red)
        # image_green = self.transforms.sobel_vertical_edge_detection(image_green)

        import utils
        utils.show_results(image_yellow, image_red, image_green)

        return [image_yellow, image_green, image_red]
