import argparse
import h5py
import itertools
import logging
import multiprocessing
import numpy as np
import sys
import tqdm

from .blockfs_stack import BlockfsStack
from blockfs.directory import Directory

SOURCE:str = None
DATASET_NAME:str = None
H5_FILE:h5py.File = None
H5_DATASET:h5py.Dataset = None
DIRECTORY:Directory = None


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",
                        help="The name of the .ims file",
                        required=True)
    parser.add_argument("--channel",
                        help="The channel # of the source to convert",
                        required=True)
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
    parser.add_argument("--voxel-size",
                        default="1.8,1.8,2.0",
                        help="The voxel size in microns, default is for 4x "
                             "SPIM. This should be three comma-separated "
                             "values, e.g. \"1.8,1.8,2.0\".")
    return parser.parse_args(args)


def check_open():
    global H5_FILE, H5_DATASET
    if not H5_DATASET:
        H5_FILE = h5py.File(SOURCE, "r")
        H5_DATASET = H5_FILE[DATASET_NAME]


def write_one(x0, y0, z0):
    check_open()
    x1 = min(x0 + 256, H5_DATASET.shape[2])
    y1 = min(y0 + 256, H5_DATASET.shape[1])
    z1 = min(z0 + 256, H5_DATASET.shape[0])
    block = H5_DATASET[z0:z1, y0:y1, x0:x1]
    for x0a, y0a, z0a in itertools.product(
            range(x0, x1, 64),
            range(y0, y1, 64),
            range(z0, z1, 64)):
        zs, ys, xs = DIRECTORY.get_block_size(x0a, y0a, z0a)
        x1a = x0a + xs
        y1a = y0a + ys
        z1a = z0a + zs
        DIRECTORY.write_block(
            block[z0a-z0:z1a-z0, y0a-y0:y1a-y0,x0a-x0:x1a-x0],
            x0a, y0a, z0a)


def write_level_1(args, stack:BlockfsStack):
    global DIRECTORY
    DIRECTORY = stack.make_l1_directory(args.n_cores)
    DIRECTORY.create()
    DIRECTORY.start_writer_processes()
    x0s = np.arange(0, stack.x_extent, 256)
    y0s = np.arange(0, stack.y_extent, 256)
    z0s = np.arange(0, stack.z_extent, 256)
    with multiprocessing.Pool(args.n_cores) as pool:
        futures = []
        for x0, y0, z0 in itertools.product(x0s, y0s, z0s):
            futures.append(pool.apply_async(
                write_one, (x0, y0, z0)))
        for future in tqdm.tqdm(futures):
            future.get()
        DIRECTORY.close()


def main(args=sys.argv[1:]):
    global SOURCE, DATASET_NAME
    args = parse_args(args)
    SOURCE=args.source
    channel_name = "Channel %s" % args.channel
    DATASET_NAME = f"/DataSet/ResolutionLevel 0/TimePoint 0/{channel_name}/Data"
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    with h5py.File(SOURCE, "r") as fd:
        ds = fd[DATASET_NAME]
        stack = BlockfsStack(ds.shape, args.dest)

    voxel_size = [int(float(_) * 1000) for _ in args.voxel_size.split(",")]
    stack.write_info_file(args.levels, voxel_size)
    write_level_1(args, stack)
    for level in range(2, args.levels+1):
        stack.write_level_n(level, n_cores=args.n_cores)


if __name__== "__main__":
    main()
