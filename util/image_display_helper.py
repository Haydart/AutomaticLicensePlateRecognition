import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt


class ImageDisplayHelper:
    mpl.rcParams['figure.dpi'] = 150
    subplot_width = None
    subplot_height = None

    def __init__(self, subplot_width, subplot_height):
        self.subplot_width = subplot_width
        self.subplot_height = subplot_height

    def subplot_image(self, image, subplot_index, title='', fix_colors=True):
        plt.subplot(self.subplot_height, self.subplot_width, subplot_index)

        if fix_colors:
            if len(image.shape) == 3 and image.shape[2] == 3:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)

        plt.title(title)
        plt.axis('off')
