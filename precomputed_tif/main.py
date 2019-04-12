import argparse
import logging
import sys
import zarr

from .stack import Stack
from .zarr_stack import ZarrStack
from .blockfs_stack import BlockfsStack

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
                        help="destination format (tiff, zarr, blockfs)",
                        default='tiff')
    parser.add_argument("--n-cores",
                        type=int,
                        default=None,
                        help="The number of cores to use for multiprocessing.")
    parser.add_argument("--log",
                        default="WARNING",
                        help="The log level for logging messages")
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    if args.format == 'zarr':
        if args.source.endswith('.tif') or args.source.endswith('.tiff'):
            stack = ZarrStack(args.source, args.dest)
            kwargs = {}
        else:
            store = zarr.NestedDirectoryStore(args.source)
            stack = ZarrStack(store, args.dest)
    elif args.format == 'blockfs':
        stack = BlockfsStack(args.source, args.dest)
    else:
        stack = Stack(args.source, args.dest)
    if args.format != 'zarr':
        if args.n_cores is None:
            kwargs = {}
        else:
            kwargs = dict(n_cores=args.n_cores)

    stack.write_info_file(args.levels)
    stack.write_level_1(**kwargs)
    for level in range(2, args.levels+1):
        stack.write_level_n(level, **kwargs)


if __name__ == "__main__":
    main()
