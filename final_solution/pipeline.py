import sys
import argparse

import final_solution.src.input_output as io
import final_solution.src.band_clipping as bc
import final_solution.src.boundings as bb
import final_solution.src.vehicles_detection as vd
import final_solution.src.heuristics as he


from copy import copy
from final_solution.src.transformation import AdvancedTransforms, BasicTransforms


class Candidates:

    def __init__(self, sobel_candidates=[], opening_candidates=[], color_candidtes=[]):
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
    image_work = copy(image)
    model = AdvancedTransforms()
    image_work = model.preprocess(image_work)

    image_sobel_method_vertical, image_sobel_method_horizontal = model.skeletonized_sobel_method(copy(image_work))
    image_opening_method = model.opening_method(copy(image_work))
    images_color_method = model.color_mask_method(copy(image))

    sobel_candidates = bc.find_candidates(bc.sobel_method, image_sobel_method_vertical, image_sobel_method_horizontal)
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


def bounding_box_filtered(image, candidates_filtered):
    image_boxes = copy(image)
    image_boxes = bb.apply_bounding_boxes(image_boxes, candidates_filtered, bb.GREEN)
    return image_boxes


def filter_heuristically(candidates):
    print("All candidates", candidates)
    candidates = he.join_separated(candidates)
    print(candidates)
    return candidates

def main(argv):
    args = parse()

    img_loader = io.ImageLoader()
    img_saver = io.ImageSaver(args.output_dir)

    vehicle_detector = vd.VehiclesDetector()
    for image in img_loader.load_images(args.input_dir):
        # for sub_image in vehicle_detector.detect_vehicles(image.image):
        #     image.image = sub_image
            candidates = process(image.image)
            # image_boxes = bounding_box(image.image, candidates)
            candidates_filtered = filter_heuristically(candidates.all)
            image_boxes = bounding_box_filtered(image.image, candidates_filtered)

            image.image = image_boxes
            img_saver.save_image(image)


if __name__ == '__main__':
    main(sys.argv)
