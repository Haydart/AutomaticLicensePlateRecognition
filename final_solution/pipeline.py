import argparse
import sys
from copy import copy

import util.band_clipping as bc
import util.boundings as bb
import util.input_output as io
from util.pipeline_transformations import TransformationPipeline


class Candidates:

    def __init__(self, sobel_candidates=None, opening_candidates=None, color_candidtes=None):
        if color_candidtes is None:
            color_candidtes = []
        if opening_candidates is None:
            opening_candidates = []
        if sobel_candidates is None:
            sobel_candidates = []
        self.sobel_candidates = sobel_candidates
        self.opening_candidates = opening_candidates
        self.color_candidates = color_candidtes

    @property
    def all(self):
        return self.sobel_candidates + self.opening_candidates + self.color_candidates


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='input_dir', required=True, type=str)
    parser.add_argument('-o', action='store', dest='output_dir', required=True, type=str)

    return parser.parse_args()


def process(image):
    working_image = copy(image)
    transformations = TransformationPipeline()
    working_image = transformations.preprocess(working_image)

    vert_sobel_image, hor_sobel_image = transformations.apply_skeletonized_sobel(copy(working_image))
    image_opening_method = transformations.apply_morph_opening(copy(working_image))
    images_color_method = transformations.apply_color_masks(copy(image))

    sobel_candidates = bc.find_candidates(bc.sobel_method, vert_sobel_image, hor_sobel_image)
    opening_candidates = bc.find_candidates(bc.opening_method, image_opening_method)

    color_candidates = []
    for image_color in images_color_method:
        try:
            candidates = bc.find_candidates(bc.color_method, image_color)
            color_candidates.extend(candidates)
        except ValueError:
            continue

    candidates = Candidates(
        sobel_candidates=sobel_candidates,
        opening_candidates=opening_candidates,
        color_candidtes=color_candidates
    )

    return candidates


def bounding_box(image, candidates):
    image_boxes = copy(image)
    image_boxes = bb.apply_bounding_boxes(image_boxes, candidates.sobel_candidates, bb.GREEN)
    image_boxes = bb.apply_bounding_boxes(image_boxes, candidates.opening_candidates, bb.RED)
    image_boxes = bb.apply_bounding_boxes(image_boxes, candidates.color_candidates, bb.BLUE)
    return image_boxes


def main(argv):
    args = parse()

    img_loader = io.ImageLoader()
    img_saver = io.ImageSaver(args.output_dir)
    for image in img_loader.load_images(args.input_dir):
        candidates = process(image.image)
        image_boxes = bounding_box(image.image, candidates)

        image.image = image_boxes
        img_saver.save_image(image)


if __name__ == '__main__':
    main(sys.argv)
