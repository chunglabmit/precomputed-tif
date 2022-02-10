import argparse
import logging
import sys
import zarr

from precomputed_tif.ngff_stack import NGFFStack
from .stack import Stack, PType
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
                        help="destination format (tiff, zarr, blockfs, ngff)",
                        default='blockfs')
    parser.add_argument("--chunk-size",
                        help="Dimensions of a precomputed chunk in x,y,z form")
    parser.add_argument("--n-cores",
                        type=int,
                        default=None,
                        help="The number of cores to use for multiprocessing.")
    parser.add_argument("--log",
                        default="WARNING",
                        help="The log level for logging messages")
    parser.add_argument("--voxel-size",
                        default="1.8,1.8,2.0",
                        help="The voxel size in microns, default is for 4x "
                             "SPIM. This should be three comma-separated "
                             "values, e.g. \"1.8,1.8,2.0\".")
    parser.add_argument("--segmentation",
                        help="If present, the volume will be saved as a segmentation instead of a 3d image",
                        action="store_true")
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    ptype = PType.SEGMENTATION if args.segmentation else PType.IMAGE
    kwargs = {}
    if args.chunk_size is not None:
        cx, cy, cz = [int(_) for _ in args.chunk_size.split(",")]
        kwargs["chunk_size"] = (cz, cy, cx)
    if args.format == 'zarr':
        if args.source.endswith('.tif') or args.source.endswith('.tiff'):
            stack = ZarrStack(args.source, args.dest, **kwargs)
            kwargs = {}
        else:
            store = zarr.NestedDirectoryStore(args.source)
            stack = ZarrStack(store, args.dest)
    elif args.format == 'blockfs':
        stack = BlockfsStack(args.source, args.dest, ptype=ptype, **kwargs)
    elif args.format == 'ngff':
        stack = NGFFStack(args.source, args.dest, **kwargs)
        stack.create()
    else:
        stack = Stack(args.source, args.dest, ptype=ptype, **kwargs)
    if args.format != 'zarr':
        if args.n_cores is None:
            kwargs = {}
        else:
            kwargs = dict(n_cores=args.n_cores)

    voxel_size = [int(float(_) * 1000) for _ in args.voxel_size.split(",")]
    stack.write_info_file(args.levels, voxel_size)
    stack.write_level_1(**kwargs)
    for level in range(2, args.levels+1):
        stack.write_level_n(level, **kwargs)


if __name__ == "__main__":
    main()
