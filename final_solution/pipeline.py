import sys
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='input_dir', required=True, type=str)
    parser.add_argument('-o', action='store', dest='output_dir', required=True, type=str)

    return parser.parse_args()


def main(argv):
    args = parse()


if __name__ == '__main__':
    main(sys.argv)
