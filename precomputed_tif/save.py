"""Save a single plane from a precomputed TIF file"""

import argparse
import numpy as np
import sys
import tifffile

from .client import get_info, read_chunk


def parse_args(args=sys.argv[1:]):
    """Parse the program arguments

    :param args:
    :return:
    """
    parser = argparse.ArgumentParser(
        description="\n".join(
            ["Save a single plane from precomputed TIF",
             "",
             "This program reads image data from a precomputed: URL and",
             "saves the data as a single .TIF plane. It can assemble the plane",
             "from any direction. The plane is taken as the two axes normal",
             "to the normal direction (e.g. to get X and Y, specify Z as the",
             "normal direction) and a max projection along the normal",
             "direction can be done.",
             "",
             "After reading and max projection, the image can be rotated",
             "and flipped. Rotations are applied, then flippings."]))
    parser.add_argument(
        "--url",
        required=True,
        help='The precomputed URL, e.g. "http://leviathan-chunglab.mit.edu"')
    parser.add_argument(
        "--output",
        required=True,
        help="The name of the output file")
    parser.add_argument(
        "--x",
        action="store_true",
        help="Take the X direction as the normal and Y+Z become the plane")
    parser.add_argument(
        "--y",
        action="store_true",
        help="Take the Y direction as the normal and X+Z become the plane")
    parser.add_argument(
        "--coordinate",
        type=int,
        required=True,
        help="The coordinate of the plane in the normal direction")
    parser.add_argument(
        "--depth",
        default=1,
        type=int,
        help="The number of images in the normal direction to combine using "
        "a max projection")
    parser.add_argument(
        "--rotate-left",
        action="store_true",
        help="Rotate the resulting image left")
    parser.add_argument(
        "--rotate-right",
        action="store_true",
        help="Rotate the resulting image right")
    parser.add_argument(
        "--rotate-180",
        action="store_true",
        help="Rotate the resulting image 180 degrees")
    parser.add_argument(
        "--flip-horizontally",
        action="store_true",
        help="Flip the image horizontally after applying rotation")
    parser.add_argument(
        "--flip-vertically",
        action="store_true",
        help="Flip the image vertically after applying rotation")
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="The mipmap level, e.g. 1, 2, 4, etc")
    parser.add_argument(
        "--format",
        default="blockfs",
        help="For a file:// url, the format of the data source")
    return parser.parse_args(args)

def main(args=sys.argv[1:]):
    options = parse_args(args)
    info = get_info(options.url)
    level = options.level
    scale = info.get_scale(level)
    # Note: scale is in X, Y, Z order. Put in numpy canonical "c" order.
    shape = scale.shape[::-1]
    if options.x:
        x0 = options.coordinate
        x1 = x0 + options.depth
        y0 = z0 = 0
        z1, y1 = shape[:-1]
        axis = 2
    elif options.y:
        y0 = options.coordinate
        y1 = y0 + options.depth
        x0 = z0 = 0
        z1, _, x1 = shape
        axis = 1
    else:
        z0 = options.coordinate
        z1 = z0 + options.depth
        x0 = y0 = 0
        y1, x1 = shape[1:]
        axis = 0
    chunk = read_chunk(options.url, x0, x1, y0, y1, z0, z1, level,
                       format=options.format)
    projection = np.max(chunk, axis)
    if options.rotate_right:
        projection = np.rot90(projection, 1)
    if options.rotate_180:
        projection = np.rot90(projection, 2)
    if options.rotate_left:
        projection = np.rot90(projection, 3)
    if options.flip_horizontally:
        projection = np.fliplr(projection)
    if options.flip_vertically:
        projection = np.flipud(projection)
    tifffile.imsave(options.output, projection)


if __name__=="__main__":
    main()