# Convert any precomputed format to NGFF format
import argparse
import itertools
import multiprocessing
import json
import sys
import tqdm
from .ngff_stack import NGFFStack
from .client import ArrayReader
from urllib.request import urlopen

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="URL of input source",
        required=True
    )
    parser.add_argument(
        "--input-format",
        help="Format of input source if a file URL, e.g. \"blockfs\".",
        default="blockfs"
    )
    parser.add_argument(
        "--output",
        help="Output directory for the NGFF volume",
        required=True
    )
    parser.add_argument(
        "--chunks",
        help="If present, override the default chunking. Format is z,y,x "
             "e.g. \"64,64,64\"."
    )
    parser.add_argument(
        "--n-workers",
        help="# of worker processes",
        default=min(multiprocessing.cpu_count(), 24)
    )
    return parser.parse_args(args)


ARRAY_READER = None
ZARR = None


def do_one(level, x0, x1, y0, y1, z0, z1):
    ZARR[0, 0, z0:z1, y0:y1, x0:x1] = ARRAY_READER[z0:z1, y0:y1, x0:x1]


def main(args=sys.argv[1:]):
    global ARRAY_READER, ZARR
    opts = parse_args(args)
    ar0 = ArrayReader(opts.input, format=opts.input_format)
    stack = NGFFStack(ar0.shape, opts.output,
                      dtype=ar0.dtype)
    if opts.chunks is not None:
        chunks = [int(_) for _ in opts.chunks.split(",")]
        stack.chunksize=chunks
    input_info_url = opts.input + "/info"
    with urlopen(input_info_url) as fd:
        input_info = json.load(fd)
    scales = input_info["scales"]
    scale0 = scales[0]
    resolution = scale0["resolution"]
    levels = len(scales)

    stack.create()
    stack.write_info_file(levels, voxel_size=resolution)
    for level in range(1, levels+1):
        ARRAY_READER = ArrayReader(
            opts.input,
            format=opts.input_format,
            level=2 ** (level - 1))
        ZARR = stack.create_dataset(level)
        with multiprocessing.Pool(opts.n_workers) as pool:
            futures = []
            for (x0, x1), (y0, y1), (z0, z1) in itertools.product(
                    zip(stack.x0(level), stack.x1(level)),
                    zip(stack.y0(level), stack.y1(level)),
                    zip(stack.z0(level), stack.z1(level))):
                futures.append(pool.apply_async(
                    do_one,
                    (level, x0, x1, y0, y1, z0, z1)))
            for future in tqdm.tqdm(futures):
                future.get()

if __name__=="__main__":
    main()