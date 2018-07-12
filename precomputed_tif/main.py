import argparse
import sys

from .stack import Stack
from .zarr_stack import ZarrStack


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
    parser.add_argument("--format",
                        type=str,
                        help="destination format (tiff, zarr)",
                        default='tiff')
    return parser.parse_args(args)


def main():
    args = parse_args()
    if args.format == 'zarr':
        stack = ZarrStack(args.source, args.dest)
    else:
        stack = Stack(args.source, args.dest)
    stack.write_info_file(args.levels)
    stack.write_level_1()  # Not needed for zarr inputs
    for level in range(2, args.levels+1):
        stack.write_level_n(level)


if __name__ == "__main__":
    main()
