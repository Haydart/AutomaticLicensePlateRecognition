import sys
import argparse
import final_solution.src.input_output as io
import final_solution.src.band_clipping as bc
import final_solution.src.boundings as bb


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

    # image_sobel_method_vertical, image_sobel_method_horizontal = model.skeletonized_sobel_method(copy(image_work))
    # image_opening_method = model.opening_method(copy(image_work))
    image_color_method = model.color_mask_method(copy(image))

    # image_sobel = BasicTransforms.sobel_vertical_edge_detection(image_color_method)
    # import utils
    # utils.show_one_image(image_sobel)

    # sobel_candidates = bc.find_candidates(bc.sobel_method, image_sobel_method_vertical, image_sobel_method_horizontal)
    # opening_candidates = bc.find_candidates(bc.opening_method, image_opening_method)
    color_candidates = bc.find_candidates(bc.color_method, image_color_method)

    candidates = Candidates(
        # sobel_candidates=sobel_candidates,
        # opening_candidates=opening_candidates,
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
