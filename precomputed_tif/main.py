import argparse
import sys

from .stack import Stack


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",
                        help="Glob expression to collect source .tiff files, "
                        "e.g. \"/path/to/img_*.tif*\"",
                        required=True)
    parser.add_argument("--dest",
                        help="Destination directory for precomputed stack.")
    parser.add_argument("--levels",
                        type=int,
                        help="# of mipmap levels",
                        default=4)
    return parser.parse_args(args)


def main():
    args = parse_args()
    stack = Stack(args.source, args.dest)
    stack.write_info_file(args.levels)
    stack.write_level_1()
    for level in range(2, args.levels+1):
        stack.write_level_n(level)


if __name__ == "__main__":
    main()