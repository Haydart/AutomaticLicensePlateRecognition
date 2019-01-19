import sys
import argparse
import final_solution.src.input_output as io
import final_solution.src.band_clipping as bc

from final_solution.src.boundings import apply_bounding_boxes

from copy import copy
from final_solution.src.transformation import AdvancedTransforms


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='input_dir', required=True, type=str)
    parser.add_argument('-o', action='store', dest='output_dir', required=True, type=str)

    return parser.parse_args()


def process(image):
    model = AdvancedTransforms()
    image = model.preprocess(image)

    image_sobel_method_vertical, image_sobel_method_horizontal = model.skeletonized_sobel_method(copy(image))
    image_opening_method = model.opening_method(copy(image))

    sobel_candidates = bc.find_candidates(bc.sobel_method, image_sobel_method_vertical, image_sobel_method_horizontal)
    opening_candidates = bc.find_candidates(bc.opening_method, image_opening_method)

    candidates = sobel_candidates + opening_candidates

    return candidates


def main(argv):
    args = parse()

    img_loader = io.ImageLoader()
    img_saver = io.ImageSaver(args.output_dir)
    for image in img_loader.load_images(args.input_dir):
        print(image.path)
        candidates = process(image.image)
        image_boxes = apply_bounding_boxes(copy(image.image), candidates)

        image.image = image_boxes
        img_saver.save_image(image)


if __name__ == '__main__':
    main(sys.argv)
