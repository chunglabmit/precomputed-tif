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

import zarr

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
        key = "_".join([str(_) for _ in level])
        for scale in self.d["scales"]:
            if scale["key"] == key:
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
            elif format == 'ngff':
                ngff_parse = urlparse(url)
                ngff_path = os.path.join(ngff_parse.netloc,
                                         unquote(ngff_parse.path))
                key = str(int(np.log2(level)))
                storage = zarr.NestedDirectoryStore(ngff_path)
                group = zarr.group(storage)
                dataset = group[key]
                dataset.read_only = True
                chunk = dataset[0, 0, z0c:z1c, y0c:y1c, x0c:x1c]
            elif format == 'zarr':
                zarr_url = url + "/" + scale.key
                zarr_parse = urlparse(zarr_url)
                zarr_path = os.path.join(zarr_parse.netloc,
                                         unquote(zarr_parse.path))
                storage = zarr.NestedDirectoryStore(zarr_path)
                dataset = zarr.Array(storage)
                chunk = dataset[z0c:z1c, y0c:y1c, x0c:x1c]
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
        result[z0c - z0:z0c - z0 + chunk.shape[0],
               y0c - y0:y0c - y0 + chunk.shape[1],
               x0c - x0:x0c - x0 + chunk.shape[2]] = chunk
    return result


class ArrayReaderBase:
    def __len__(self):
        return self.shape[0]

    @property
    def shape(self):
        raise NotImplementedError("Implement shape in your class")

    def __getitem__(self, key):
        def s(idx, axis):
            if idx is None:
                return 0
            if idx < 0:
                return self.shape[axis] + idx
            return idx

        def e(idx, axis):
            if idx is None:
                return self.shape[axis]
            if idx < 0:
                return self.shape[axis] + idx
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
        block = self.read_chunk(x0, x1, y0, y1, z0, z1)[::zs, ::ys, ::xs]
        if xsquish:
            block = block[:, :, 0]
        if ysquish:
            block = block[:, 0]
        if zsquish:
            block = block[0]
        return block

    def read_chunk(self, x0, x1, y0, y1, z0, z1):
        raise NotImplementedError("Implement read_chunk in your class")


class ArrayReader(ArrayReaderBase):
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

    @property
    def shape(self):
        return self.scale.shape[::-1]

    @property
    def dtype(self):
        return self.info.data_type

    def read_chunk(self, x0, x1, y0, y1, z0, z1):
        return read_chunk(self.url, x0, x1, y0, y1, z0, z1,
                          self.level, self.format)



class DANDIArrayReader(ArrayReaderBase):

    CHUNK_TRANSFORM_MATRIX_KWD = "ChunkTransformMatrix"
    def __init__(self, urls, level=1):
        self.level = level
        self.urls = urls
        self.array_readers = [ArrayReader(url, format='ngff', level=level)
                              for url in urls]
        self.offsets = []
        for url in self.urls:
            # Name is something like foo_spim.ngff
            #
            # Modern form is to extract offsets from affine transform
            # stored in the metadata sidecar
            #
            sidecar_url = url[:-4] + "json"
            with urlopen(sidecar_url) as fd:
                sidecar = json.load(fd)
            if self.CHUNK_TRANSFORM_MATRIX_KWD in sidecar:
                matrix = sidecar[self.CHUNK_TRANSFORM_MATRIX_KWD]
                self.offsets.append((int(matrix[0][-1] // level),
                                     int(matrix[1][-1] // level),
                                     int(matrix[2][-1] // level)))
            else:
                xform_url = url[:-9] + "transforms.json"
                with urlopen(xform_url) as fd:
                    xform = json.load(fd)
                self.offsets.append(
                    tuple(int(xform[0]["TransformationParameters"][_]) // level
                          for _ in ("ZOffset", "YOffset", "XOffset")))

    def x0(self, idx):
        return self.offsets[idx][2]

    def x1(self, idx):
        return self.x0(idx) + self.array_readers[idx].shape[2]

    def y0(self, idx):
        return self.offsets[idx][1]

    def y1(self, idx):
        return self.y0(idx) + self.array_readers[idx].shape[1]

    def z0(self, idx):
        return self.offsets[idx][0]

    def z1(self, idx):
        return self.z0(idx) + self.array_readers[idx].shape[0]

    @property
    def shape(self):
        return tuple(np.max([fn(_) for _ in range(len(self.urls))])
                     for fn in (self.z1, self.y1, self.x1))

    @property
    def dtype(self):
        return self.array_readers[0].dtype


    def get(self,
            idx:int,
            x0:int, x1:int, y0:int, y1:int, z0:int, z1:int):
        """
        Get a data range from a chunk.
        :param idx: the index of the chunk in question
        :param x0: The start x in global coordinates
        :param x1: The end x in global coordinates
        :param y0: The start y in global coordinates
        :param y1: The end y in global coordinates
        :param z0: The start z in global coordinates
        :param z1: The end z in global coordinates
        :return: a two-tuple of an array containing the data read and
                 an array similarly sized to data, giving the manhattan
                 distance to the nearest edge.
        """
        data = np.zeros((z1-z0, y1-y0, x1-x0), self.dtype)
        ar = self.array_readers[idx]
        x0c = self.x0(idx)
        y0c = self.y0(idx)
        z0c = self.z0(idx)
        x1c = self.x1(idx)
        y1c = self.y1(idx)
        z1c = self.z1(idx)
        if x0c >= x1 or x0 >= x1c or \
           y0c >= y1 or y0 >= y1c or \
           z0c >= z1 or z0 >= z1c:
           return

        x0a = max(x0c, x0)
        x1a = min(x1c, x1)
        y0a = max(y0c, y0)
        y1a = min(y1c, y1)
        z0a = max(z0c, z0)
        z1a = min(z1c, z1)
        chunk = ar[z0a-z0c:z1a-z0c, y0a-y0c:y1a-y0c, x0a-x0c:x1a - x0c]
        data[z0a-z0:z1a-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] += chunk
        nx = np.arange(x0-x0c, x1-x0c)
        fx = np.arange(x1c-x0-1, x1c-x1-1, -1)
        ny = np.arange(y0-y0c, y1-y0c)
        fy = np.arange(y1c-y0-1, y1c-y1-1, -1)
        nz = np.arange(z0-z0c, z1-z0c)
        fz = np.arange(z1c-z0-1, z1c-z1-1, -1)
        xd = np.minimum(nx.reshape(1, 1, -1), fx.reshape(1, 1, -1))
        yd = np.minimum(ny.reshape(1, -1, 1), fy.reshape(1, -1, 1))
        zd = np.minimum(nz.reshape(-1, 1, 1), fz.reshape(-1, 1, 1))
        return data, np.minimum(xd, np.minimum(yd, zd))


    def read_chunk(self, x0, x1, y0, y1, z0, z1):
        datas = []
        distances = []
        #
        # We do cosine blending between the two pixels furthest away
        # from their edges. The furthest from the edge is "a" and the
        # second-furthest is "b"
        #
        a = np.zeros((z1-z0, y1-y0, x1-x0), np.uint8)
        ad = -np.ones_like(a, dtype=np.int32)
        b = np.zeros_like(a)
        bd = -np.ones_like(a, dtype=np.int32)
        i = 0
        for idx in range(len(self.urls)):
            result = self.get(idx, x0, x1, y0, y1, z0, z1)
            if result is None:
                continue
            data, distance = result
            datas.append(data)
            distances.append(distance)
            a_mask = distance >= np.maximum(0, ad)
            b[a_mask]  = a[a_mask]
            bd[a_mask] = ad[a_mask]
            a[a_mask] = i
            ad[a_mask] = distance[a_mask]
            b_mask = (~ a_mask) & (distance >= np.maximum(0, bd))
            b[b_mask] = i
            bd[b_mask] = distance[b_mask]
            i += 1
        if i == 0:
            return np.zeros((z1-z0, y1-y0, x1-x0), self.dtype)
        datas = np.stack(datas)
        distances = np.stack(distances)
        double_z, double_y, double_x = np.where(bd >= 0)
        single_z, single_y, single_x = np.where((ad >= 0) & (bd < 0))
        result = np.zeros_like(a, dtype=self.dtype)
        result[single_z, single_y, single_x] = \
            datas[a[single_z, single_y, single_x], single_z, single_y, single_x]
        if len(double_z) > 0:
            a_idx = a[double_z, double_y, double_x]
            distances_a = distances[a_idx, double_z, double_y, double_x]
            b_idx = b[double_z, double_y, double_x]
            distances_b = distances[b_idx, double_z, double_y, double_x]
            datas_a = datas[a_idx, double_z, double_y, double_x]
            datas_b = datas[b_idx, double_z, double_y, double_x]
            angle = np.arctan2(distances_a, distances_b)
            result[double_z, double_y, double_x] = \
                np.sin(angle) ** 2 * datas_a + np.cos(angle) ** 2 * datas_b
        return result


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
