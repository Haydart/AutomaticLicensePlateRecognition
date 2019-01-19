import cv2


class BasicTransforms:

    @staticmethod
    def gray_scale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def bilateral_filter(image):
        return cv2.bilateralFilter(image, 32, 40, 40)




class Model:

    def __init__(self, transforms=BasicTransforms):
        self.transforms = transforms

    def preproces(self, image):
        image = self.transforms.gray_scale(image)
        image = self.transforms.bilateral_filter(image)
        return image
