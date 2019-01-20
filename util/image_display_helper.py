import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt


class ImageDisplayHelper:
    mpl.rcParams['figure.dpi'] = 150
    subplot_width = None
    subplot_height = None
    subplot_index = 0
    pipeline_debug_enabled = False

    def __init__(self, debug_pipeline, subplot_width, subplot_height):
        self.subplot_width = subplot_width
        self.subplot_height = subplot_height
        self.pipeline_debug_enabled = debug_pipeline

    def add_to_plot(self, image, subplot_index=None, title='', fix_colors=True):
        if self.pipeline_debug_enabled:
            current_subplot_index = None
            print(subplot_index)
            if not subplot_index:
                self.subplot_index = self.subplot_index + 1
                current_subplot_index = self.subplot_index
            else:
                current_subplot_index = subplot_index

            plt.subplot(self.subplot_height, self.subplot_width, current_subplot_index)

            if fix_colors:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image)

            plt.title(title)
            plt.axis('off')

    def plot_results(self):
        if self.pipeline_debug_enabled:
        plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
        fig = plt.gcf()
        fig.set_size_inches(10, 15)
        print('plot display')
        plt.show()

    def save_results(self, path):
        plt.savefig(path)
