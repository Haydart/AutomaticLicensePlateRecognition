from copy import copy

from util.basic_transformations import BasicTransformations
from util.image_display_helper import ImageDisplayHelper


class PipelineTransformations:
    basic_transformations = None
    pipeline_debug_enabled = False

    def __init__(self, debug_pipeline):
        self.pipeline_debug_enabled = debug_pipeline
        self.basic_transformations = BasicTransformations(ImageDisplayHelper(debug_pipeline, 2, 5))

    def preprocess(self, image):
        image = self.basic_transformations.gray_scale(image)
        image = self.basic_transformations.bilateral_filter(image)
        return image

    def apply_skeletonized_sobel(self, image):
        image_vertical = self.basic_transformations.sobel_vertical_edge_detection(copy(image))
        image_horizontal = self.basic_transformations.sobel_horizontal_edge_detection(copy(image))

        image_vertical = self.basic_transformations.skeletonize(image_vertical)
        image_horizontal = self.basic_transformations.skeletonize(image_horizontal)

        return image_vertical, image_horizontal

    def apply_morph_opening(self, image):
        image = self.basic_transformations.histogram_equalization(image)
        image = self.basic_transformations.morphological_opening(image)
        image = self.basic_transformations.binary_threshold(image, 100)
        return image

    def apply_color_masks(self, image):
        image_yellow = self.basic_transformations.color_mask(copy(image), 'yellow')
        image_red = self.basic_transformations.color_mask(copy(image), 'red')
        image_green = self.basic_transformations.color_mask(copy(image), 'green')
        # image = self.transforms.color_mask(copy(image), 'blue')

        # image_yellow = self.transforms.sobel_vertical_edge_detection(image_yellow)
        # image_red = self.transforms.sobel_vertical_edge_detection(image_red)
        # image_green = self.transforms.sobel_vertical_edge_detection(image_green)

        from util import utils
        utils.show_results(image_yellow, image_red, image_green)

        return [image_yellow, image_green, image_red]
