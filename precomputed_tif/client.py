"""Client for downloading image data

"""

import itertools
import json
from urllib.request import urlopen, urlparse
from urllib.parse import unquote
import numpy as np
import os
import tifffile
import time

__cache = {}

class Scale:
    """Represents a mipmap level on the precomputed data source"""

    def __init__(self, d):
        self.d = d

    @property
    def chunk_sizes(self):
        """The dimensions of one chunk"""
        return self.d["chunk_sizes"][0]

    @property
    def encoding(self):
        return self.d["encoding"]

    @property
    def key(self):
        """The key for data retrieval"""
        return self.d["key"]

    @property
    def shape(self):
        """The voxel dimensions (the "size" field)"""
        return self.d["size"]

    @property
    def offset(self):
        """The offset of the dataset in the larger space in voxels"""
        return self.d["voxel_offset"]


class Info:
    """information on the precomputed data source"""

    TYPE_INFO = "info"
    TYPE_SEGMENTATION = "segmentation"

    def __init__(self, d):
        self.d = d

    @property
    def data_type(self):
        """The data type of the data source, e.g. np.uint16"""
        return np.dtype(self.d["data_type"])

    @property
    def type(self):
        """The type of the voxel data

        One of "image", or "segmentation"
        """
        return self.d["type"]

    def get_scale(self, level) -> Scale:
        """Get the Scale object for the given level

        :param level: either a single level, e.g. 1 or a tuple of the
               mipmap levels for X, Y and Z
        """
        if np.isscalar(level):
            level = (level, level, level)
        else:
            level = tuple(level)
        for scale in self.d["scales"]:
            if tuple(scale["resolution"]) == level:
                return Scale(scale)
        else:
            raise KeyError("No such level: %s" % str(level))


def get_info(url) -> Info:
    """Get the info file for a precomputed URL

    :param url: The precomputed URL
    :returns: the info for the precomputed data source at the URL
    """
    precomputed = "precomputed://"
    if url.startswith(precomputed):
        url = url[len(precomputed):]
    if url in __cache:
        return __cache[url]
    info_url = url + "/info"
    response = urlopen(info_url)
    __cache[url] = Info(json.loads(response.read().decode("ascii")))
    return __cache[url]


def clear_cache(url=None):
    """Clear the cache of info files

    :param url: url to clear or None (default) for all
    """
    if url is None:
        __cache.clear()
    elif url in __cache:
        del __cache[url]


def _chunk_start(coord, offset, stride):
    if coord < offset:
        return offset
    modulo = (coord - offset) % stride
    return coord - modulo


def _chunk_end(coord, offset, stride, end):
    result = _chunk_start(coord, offset, stride) + stride
    if result > end:
        return end
    return result


def read_chunk(url, x0, x1, y0, y1, z0, z1, level=1, format="tiff"):
    """Read an arbitrary chunk of data

    :param url: Base URL of the precomputed data source
    :param x0: starting X coordinate, in the level's coordinate space
    :param x1: ending X coordinate (non-inclusive)
    :param y0: starting Y coordinate
    :param y1: ending Y cooridinate
    :param z0: starting Z coordinate
    :param z1: ending Z coordinate
    :param level: mipmap level
    :param format: the read format if it's a file URL. Defaults to tiff, but
    you can use "blockfs"
    :return: a Numpy array containing the data
    """
    is_file = urlparse(url).scheme.lower() == "file"
    info = get_info(url)
    scale = info.get_scale(level)
    result = np.zeros((z1-z0, y1-y0, x1-x0), info.data_type)
    shape = np.array(scale.shape)
    offset = np.array(scale.offset)
    stride = np.array(scale.chunk_sizes)
    end = offset + shape

    x0d = _chunk_start(x0, offset[0], stride[0])
    x1d = _chunk_end(x1, offset[0], stride[0], end[0])
    y0d = _chunk_start(y0, offset[1], stride[1])
    y1d = _chunk_end(y1, offset[1], stride[1], end[1])
    z0d = _chunk_start(z0, offset[2], stride[2])
    z1d = _chunk_end(z1, offset[2], stride[2], end[2])
    for x0c, y0c, z0c in itertools.product(
        range(x0d, x1d, stride[0]),
        range(y0d, y1d, stride[1]),
        range(z0d, z1d, stride[2])):
        x1c = min(x1d, x0c + stride[0])
        y1c = min(y1d, y0c + stride[1])
        z1c = min(z1d, z0c + stride[2])
        chunk_url = url + "/" + scale.key + "/%d-%d_%d-%d_%d-%d" % (
            x0c, x1c, y0c, y1c, z0c, z1c)
        if is_file:
            if format == "tiff":
                chunk_url += ".tiff"
                with urlopen(chunk_url) as fd:
                    chunk = tifffile.imread(fd)
            elif format == "blockfs":
                from blockfs import Directory
                from .blockfs_stack import BlockfsStack
                directory_url = url + "/" + scale.key + "/" +\
                                BlockfsStack.DIRECTORY_FILENAME
                directory_parse = urlparse(directory_url)
                directory_path = os.path.join(
                    directory_parse.netloc,
                    unquote(directory_parse.path))
                directory = Directory.open(directory_path)
                chunk = directory.read_block(x0c, y0c, z0c)
            else:
                raise NotImplementedError("Can't read %s yet" % format)
        else:
            response = urlopen(chunk_url)
            data = response.read()
            chunk = np.frombuffer(data, info.data_type).reshape(
                    (z1c - z0c, y1c - y0c, x1c - x0c))
        if z0c < z0:
            chunk = chunk[z0 - z0c:]
            z0c = z0
        if z1c > z1:
            chunk = chunk[:z1-z0c]
            z1c = z1
        if y0c < y0:
            chunk = chunk[:, y0 - y0c:]
            y0c = y0
        if y1c > y1:
            chunk = chunk[:, :y1-y0c]
            y1c = y1
        if x0c < x0:
            chunk = chunk[:, :, x0 - x0c:]
            x0c = x0
        if x1c > x1:
            chunk = chunk[:, :, :x1-x0c]
            x1c = x1
        result[z0c - z0:z1c - z0,
               y0c - y0:y1c - y0,
               x0c - x0:x1c - x0] = chunk
    return result


class ArrayReader:
    def __init__(self, url, format='tiff', level=1):
        """
        Initialize the reader with the precomputed data source URL
        :param url: URL of the data source
        :param format: either 'tiff', 'blockfs' or 'zarr'
        :param level: the mipmap level
        """
        self.url = url
        self.format = format
        self.level = level
        self.info = get_info(url)
        self.scale = self.info.get_scale(level)

    def __len__(self):
        return self.scale.shape[-1]

    @property
    def shape(self):
        return self.scale.shape[::-1]

    @property
    def dtype(self):
        return self.info.data_type

    def __getitem__(self, key):
        def s(idx, axis):
            if idx is None:
                return 0
            if idx < 0:
                return self.scale.shape[2-axis] + idx
            return idx

        def e(idx, axis):
            if idx is None:
                return self.scale.shape[2-axis]
            if idx < 0:
                return self.scale.shape[2 - axis] + idx
            return idx

        assert len(key) == 3, "Please specify 3 axes when indexing"
        if isinstance(key[0], slice):
            z0 = s(key[0].start, 0)
            z1 = e(key[0].stop, 0)
            zs = key[0].step
            zsquish=False
        else:
            z0 = s(key[0], 0)
            z1 = z0 + 1
            zsquish=True
            zs = 1
        if isinstance(key[1], slice):
            y0 = s(key[1].start, 1)
            y1 = e(key[1].stop, 1)
            ys = key[1].step
            ysquish=False
        else:
            y0 = s(key[1], 1)
            y1 = y0 + 1
            ysquish=True
            ys = 1
        if isinstance(key[2], slice):
            x0 = s(key[2].start, 2)
            x1 = e(key[2].stop, 2)
            xs = key[2].step
            xsquish=False
        else:
            x0 = s(key[2], 2)
            x1 = x0 + 1
            xsquish=True
            xs = 1
        block = read_chunk(self.url, x0, x1, y0, y1, z0, z1,
                           self.level, self.format)[::zs, ::ys, ::xs]
        if xsquish:
            block = block[:, :, 0]
        if ysquish:
            block = block[:, 0]
        if zsquish:
            block = block[0]
        return block


if __name__ == "__main__":
    import neuroglancer
    from nuggt.utils.ngutils import layer, gray_shader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--x0", type=int)
    parser.add_argument("--x1", type=int)
    parser.add_argument("--y0", type=int)
    parser.add_argument("--y1", type=int)
    parser.add_argument("--z0", type=int)
    parser.add_argument("--z1", type=int)
    args = parser.parse_args()
    data = read_chunk(args.url, args.x0, args.x1, args.y0, args.y1,
                      args.z0, args.z1 )
    viewer = neuroglancer.Viewer()
    print(viewer)
    with viewer.txn() as txn:
        layer(txn, "image", data, gray_shader, 1.0)
    time.sleep(100000)