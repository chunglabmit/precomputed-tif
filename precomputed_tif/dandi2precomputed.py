import argparse
import itertools
import logging
import multiprocessing
import pathlib
import numpy as np
import sys
import tqdm

from .blockfs_stack import BlockfsStack
from blockfs.directory import Directory
from .client import DANDIArrayReader

SOURCE:DANDIArrayReader = None
DIRECTORY:Directory = None


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("ngffs",
                        help="The NGFF files that comprise the volume",
                        nargs="+")
    parser.add_argument("--dest",
                        help="Destination directory for precomputed stack.")
    parser.add_argument("--levels",
                        type=int,
                        help="# of mipmap levels",
                        default=4)
    parser.add_argument("--n-cores",
                        type=int,
                        default=None,
                        help="The number of cores to use for multiprocessing.")
    parser.add_argument("--log",
                        default="WARNING",
                        help="The log level for logging messages")
    return parser.parse_args(args)


def write_one(x0, y0, z0):
    x1, y1, z1 = [a0 + s for a0, s in
                  zip((x0, y0, z0),DIRECTORY.get_block_size(x0, y0, z0))]
    block = SOURCE[z0:z1, y0:y1, x0:x1]
    DIRECTORY.write_block(block, x0, y0, z0)


def write_level_1(args, stack:BlockfsStack):
    global DIRECTORY
    DIRECTORY = stack.make_l1_directory(args.n_cores)
    DIRECTORY.create()
    DIRECTORY.start_writer_processes()
    x0s = np.arange(0, stack.x_extent, 64)
    y0s = np.arange(0, stack.y_extent, 64)
    z0s = np.arange(0, stack.z_extent, 64)
    with multiprocessing.Pool(args.n_cores) as pool:
        futures = []
        for x0, y0, z0 in itertools.product(x0s, y0s, z0s):
            futures.append(pool.apply_async(
                write_one, (x0, y0, z0)))
        for future in tqdm.tqdm(futures):
            future.get()
        DIRECTORY.close()


def main(args=sys.argv[1:]):
    global SOURCE
    opts = parse_args(args)
    urls = [pathlib.Path(ngff).as_uri() for ngff in opts.ngffs]
    SOURCE=DANDIArrayReader(urls)
    stack = BlockfsStack(SOURCE.shape, opts.dest)
    logging.basicConfig(level=getattr(logging, opts.log.upper()))
    dandi_info = SOURCE.get_info()
    voxel_size = dandi_info["scales"][0]["resolution"]
    stack.write_info_file(opts.levels, voxel_size)
    write_level_1(opts, stack)
    for level in range(2, opts.levels+1):
        stack.write_level_n(level, n_cores=opts.n_cores)


if __name__== "__main__":
    main()
