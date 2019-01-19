import cv2
import numpy as np


class BasicTransforms:

    @staticmethod
    def gray_scale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def bilateral_filter(image):
        return cv2.bilateralFilter(image, 32, 40, 40)

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
    def skeletonization(image):
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



class Model:

    def __init__(self, transforms=BasicTransforms):
        self.transforms = transforms

    def preprocess(self, image):
        image = self.transforms.gray_scale(image)
        image = self.transforms.bilateral_filter(image)
        return image

    def skeletonized_sobel_method(self, image):
        image = self.transforms.sobel_vertical_edge_detection(image)
        image = self.transforms.skeletonization(image)
        return image
