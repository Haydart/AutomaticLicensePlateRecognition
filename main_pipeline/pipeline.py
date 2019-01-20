import argparse
import sys
from copy import copy

import util.band_clipping as bc
import util.bounding_boxes as bb
import util.input_output as io
from main_pipeline.candidates import Candidates
from util.pipeline_transformations import PipelineTransformations


def main(argv):
    args = parse()

    img_loader = io.ImageLoader()
    img_saver = io.ImageSaver(args.output_dir)

    for image in img_loader.load_images(args.input_dir):
        candidates = process(image.image)
        image_boxes = apply_bounding_boxex(image.image, candidates)

        image.image = image_boxes
        img_saver.save_image(image)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='input_dir', required=True, type=str)
    parser.add_argument('-o', action='store', dest='output_dir', required=True, type=str)

    return parser.parse_args()


def process(image):
    transformations = PipelineTransformations(debug_pipeline=True)
    working_image = copy(image)
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
        color_candidates=color_candidates
    )

    return candidates


def apply_bounding_boxex(image, candidates):
    image_boxes = copy(image)
    image_boxes = bb.apply_bounding_boxes(image_boxes, candidates.sobel_candidates, bb.GREEN)
    image_boxes = bb.apply_bounding_boxes(image_boxes, candidates.opening_candidates, bb.RED)
    image_boxes = bb.apply_bounding_boxes(image_boxes, candidates.color_candidates, bb.BLUE)
    return image_boxes


if __name__ == '__main__':
    main(sys.argv)
