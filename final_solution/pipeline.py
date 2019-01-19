import sys
import argparse
import final_solution.input_output as io

from copy import copy
from final_solution.transformation import Model


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='input_dir', required=True, type=str)
    parser.add_argument('-o', action='store', dest='output_dir', required=True, type=str)

    return parser.parse_args()


def process(image):
    model = Model()
    image = model.preprocess(image)

    image_sobel_method = model.skeletonized_sobel_method(copy(image))

    return image_sobel_method


def main(argv):
    args = parse()

    img_loader = io.ImageLoader()
    for image in img_loader.load_images(args.input_dir):
        image = process(image)
        import utils
        utils.show_results(image,image,image,image)



if __name__ == '__main__':
    main(sys.argv)
