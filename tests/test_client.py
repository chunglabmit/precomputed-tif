import contextlib
import json
import pathlib
import shutil
import tempfile
import unittest
import unittest.mock
import urllib.request
import numpy as np
import precomputed_tif.client
from precomputed_tif import ZarrStack
from precomputed_tif.blockfs_stack import BlockfsStack
from precomputed_tif.client import read_chunk, clear_cache, ArrayReader
from precomputed_tif.client import get_ngff_info
from precomputed_tif.client import DANDIArrayReader
from precomputed_tif.ngff_stack import NGFFStack
from precomputed_tif.stack import StackBase, Stack
from precomputed_tif.utils import make_case


class MockResponse:

    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


class MockUrlOpen:
    """A class that mimics urllib.request.urlopen"""

    BASE_URL = "mock://"

    def __init__(self):
        self.info = {
  "data_type": "uint16",
  "mesh": "mesh",
  "num_channels": 1,
  "scales": [
    {
      "chunk_sizes": [
        [
          64,
          64,
          64
        ]
      ],
      "encoding": "raw",
      "key": "1_1_1",
      "resolution": [
        1,
        1,
        1
      ],
      "size": [
        128,
        128,
        128
      ],
      "voxel_offset": [
        0,
        0,
        0
      ]
    }
        ]
        }
        self.data = np.random.RandomState(1234)\
            .randint(0, 4095, (128, 128, 128), dtype=np.uint16)

    def __call__(self, url):
        if url == self.BASE_URL + "/info":
            return MockResponse(json.dumps(self.info).encode("ascii"))
        base, directory, filename = url.rsplit("/", 2)
        assert(base == self.BASE_URL)
        for scale in self.info["scales"]:
            if scale["key"] == directory:
                (x0, x1), (y0, y1), (z0, z1) = [
                    map(int, _.split("-")) for _ in filename.split("_")]
                xoff = scale["voxel_offset"][0]
                yoff = scale["voxel_offset"][1]
                zoff = scale["voxel_offset"][2]
                xscale = scale["size"][0]
                yscale = scale["size"][1]
                zscale = scale["size"][2]
                xcs = scale["chunk_sizes"][0][0]
                ycs = scale["chunk_sizes"][0][1]
                zcs = scale["chunk_sizes"][0][2]
                assert(x0 >= xoff)
                assert(x1 <= xoff + xscale)
                assert(y0 >= yoff)
                assert(y1 <= yoff + yscale)
                assert(z0 >= zoff)
                assert(z1 <= zoff + zscale)
                assert((x0 - xoff)
                       % xcs == 0)
                assert((y0 - yoff)
                       % ycs == 0)
                assert((z0 - zoff)
                       % zcs == 0)
                assert((x1 - x0 == xcs) or
                       (x1 == xoff + xscale and
                        x0 > xoff + xscale -
                        xcs))
                assert((y1 - y0 == ycs) or
                       (y1 == yoff + yscale and
                        y0 > yoff + yscale -
                        ycs))
                assert((z1 - z0 == zcs) or
                       (z1 == zoff + zscale and
                        z0 > zoff + zscale -
                        zcs))
                chunk = self.data[z0-zoff:z1-zoff,
                                  y0-yoff:y1-yoff,
                                  x0-xoff:x1-xoff]
                return MockResponse(np.ascontiguousarray(chunk).data)

class TestClient(unittest.TestCase):

    def setUp(self):
        clear_cache(MockUrlOpen.BASE_URL)

    def test_read_all(self):
        with unittest.mock.patch(
                "precomputed_tif.client.urlopen",
                MockUrlOpen()) as mock:
            data = read_chunk(mock.BASE_URL, 0, 128, 0, 128, 0, 128)
            np.testing.assert_array_equal(data, mock.data)

    def test_read_across_blocks(self):
        with unittest.mock.patch(
                "precomputed_tif.client.urlopen",
                MockUrlOpen()) as mock:
            data = read_chunk(mock.BASE_URL, 32, 96, 16, 80, 8, 72)
            np.testing.assert_array_equal(data, mock.data[8:72, 16:80, 32:96])

    def test_offset(self):
        with unittest.mock.patch(
                "precomputed_tif.client.urlopen",
                MockUrlOpen()) as mock:
            mock.info["scales"][0]["voxel_offset"] = (10, 20, 30)
            data = read_chunk(mock.BASE_URL, 30, 60, 30, 60, 30, 60)
            np.testing.assert_array_equal(data, mock.data[:30, 10:40, 20:50])

    def test_different_sizes(self):
        with unittest.mock.patch(
                "precomputed_tif.client.urlopen",
                MockUrlOpen()) as mock:
            mock.info["scales"][0]["size"] = (96, 120, 128)
            data = read_chunk(mock.BASE_URL, 0, 96, 0, 120, 0, 128)
            np.testing.assert_array_equal(data, mock.data[:128, :120, :96])

    def test_array_reader(self):
        with unittest.mock.patch(
                "precomputed_tif.client.urlopen",
                MockUrlOpen()) as mock:
            mock.info["scales"][0]["size"] = (128, 128, 128)
            a = ArrayReader(mock.BASE_URL)
            data = a[10:20, 30:40, 50:60]
            np.testing.assert_array_equal(data, mock.data[10:20, 30:40, 50:60])
            data = a[10:20:2, 30:40:2, 50:60:2]
            np.testing.assert_array_equal(
                data, mock.data[10:20:2, 30:40:2, 50:60:2])
            data = a[10, 30, 50]
            self.assertEqual(data, mock.data[10, 30, 50])
            data = a[:10, :10, :10]
            np.testing.assert_array_equal(data, mock.data[:10, :10, :10])
            data = a[-10:, -10:, -10:]
            np.testing.assert_array_equal(data, mock.data[-10:, -10:, -10:])
            data = a[10:-108, 30:-88, 50:-68]
            np.testing.assert_array_equal(
                data, mock.data[10:-108, 30:-88, 50:-68])

    def teesstt_file_array_reader(self, format, klass:StackBase):
        with make_case(np.uint16, (100, 201, 300), klass=klass)\
                as (stack, npstack):
            if isinstance(stack, NGFFStack):
                stack.create()
            stack.write_info_file(2)
            stack.write_level_1()
            stack.write_level_n(2)
            url = pathlib.Path(stack.dest).as_uri()
            ar = ArrayReader(url, format=format)
            a = ar[:, :, :]
            np.testing.assert_array_equal(a, npstack)
            ar = ArrayReader(url, format=format, level=2)
            value = int(ar[0, 0, 0])
            expected = np.mean(npstack[:2, :2, :2])
            self.assertLess(abs(value - expected), 1)

    def test_blockfs(self):
        self.teesstt_file_array_reader("blockfs", BlockfsStack)

    def test_tiff(self):
        self.teesstt_file_array_reader("tiff", Stack)

    def test_zarr(self):
        self.teesstt_file_array_reader("zarr", ZarrStack)

    def test_ngff(self):
        self.teesstt_file_array_reader("ngff", NGFFStack)

    def test_ngff_info(self):
        with make_case(np.uint16, (100, 201, 300), klass=NGFFStack,
                       chunk_size=(64, 64, 64)) \
                as (stack, npstack):
            stack.create()
            stack.write_info_file(2)
            stack.write_level_1()
            stack.write_level_n(2)

            url = pathlib.Path(stack.dest).as_uri()
            info = get_ngff_info(url)
            scale = info.get_scale(1)
            self.assertSequenceEqual(scale.shape, (300, 201, 100))
            chunk_sizes = scale.chunk_sizes
            self.assertSequenceEqual(chunk_sizes, (64, 64, 64))

def make_bids_transform(xoff, yoff, zoff):
    return [dict(
        SourceReferenceFrame="original",
        TargetReferenceFrame="stitched",
        TransformationType="translation-3d",
        TransformationParameters= {
                "XOffset": xoff,
                "YOffset": yoff,
                "ZOffset": zoff
            }
        )]

def make_sidecar(xoff, yoff, zoff, voxel_size):
    return dict(
        PixelSize=voxel_size,
        ChunkTransformMatrix=[
        [voxel_size[2], 0., 0., xoff],
        [0., voxel_size[1], 0., yoff],
        [0., 0., voxel_size[0], zoff],
        [0., 0., 0., 1.0]
    ],
    ChunkTransformMatrixAxis = ["X", "Y", "Z"])

@contextlib.contextmanager
def make_dandi_case(y_offset,
                   old=True,
                   voxel_size=[2.564, 3.625, 2.564],
                   levels=1):
    with make_case(np.uint16, (100, 200, 300),
                   klass=NGFFStack,
                   destname="chunk1_spim.ngff") as (stack1, volume1):
        stack1.create()
        stack1.write_info_file(levels)
        stack1.write_level_1()
        for level in range(2, levels + 1):
            stack1.write_level_n(level)
        with make_case(np.uint16, (100, 200, 300),
                       klass=NGFFStack,
                       destname="chunk2_spim.ngff") as (stack2, volume2):
            stack2.create()
            stack2.write_info_file(levels)
            stack2.write_level_1()
            for level in range(2, levels+1):
                stack2.write_level_n(level)
            dest1 = pathlib.Path(stack1.dest)
            url1 = dest1.as_uri()
            dest2 = pathlib.Path(stack2.dest)
            url2 = dest2.as_uri()
            sidecar_path1 = dest1.parent / (dest1.stem + ".json")
            sidecar_path2 = dest2.parent / (dest2.stem + ".json")
            if old:
                xform_path1 = dest1.parent / (dest1.stem[:-4] + "transforms.json")
                xform_path2 = dest2.parent / (dest2.stem[:-4] + "transforms.json")
                with xform_path1.open("w") as fd:
                    json.dump(make_bids_transform(0, 0, 0), fd, indent=2)
                with xform_path2.open("w") as fd:
                    json.dump(make_bids_transform(0, y_offset, 0), fd, indent=2)
                with sidecar_path1.open("w") as fd:
                    json.dump({}, fd)
                with sidecar_path2.open("w") as fd:
                    json.dump({}, fd)
            else:
                with sidecar_path1.open("w") as fd:
                    json.dump(make_sidecar(0, 0, 0, voxel_size), fd)
                with sidecar_path2.open("w") as fd:
                    json.dump(make_sidecar(0, y_offset, 0, voxel_size), fd)
            yield (url1, volume1), (url2, volume2)


class TestDandi(unittest.TestCase):
    def test_single(self):
        with make_dandi_case(100) as ((url1, volume1), (url2, volume2)):
            ar = DANDIArrayReader([url1, url2])
            self.assertSequenceEqual((100, 300, 300), ar.shape)
            np.testing.assert_array_equal(ar[:10, :10, :10],
                                          volume1[:10, :10, :10])
            np.testing.assert_array_equal(ar[:10, 290:, :10],
                                          volume2[:10, -10:, :10])

    def test_double(self):
        with make_dandi_case(100) as ((url1, volume1), (url2, volume2)):
            ar = DANDIArrayReader([url1, url2])
            middle = ar[:10, 100:200, :10]
            bottom = volume1[:10, 100:, :10]
            top = volume2[:10, :100, :10]
            minval = np.minimum(bottom, top)
            maxval = np.maximum(bottom, top)
            self.assertTrue(np.all((middle >= minval) & (middle <= maxval)))

    def test_sidecar(self):
        with make_dandi_case(100, True) as ((url1, volume1), (url2, volume2)):
            ar = DANDIArrayReader([url1, url2])
            middle = ar[:10, 100:200, :10]
            bottom = volume1[:10, 100:, :10]
            top = volume2[:10, :100, :10]
            minval = np.minimum(bottom, top)
            maxval = np.maximum(bottom, top)
            self.assertTrue(np.all((middle >= minval) & (middle <= maxval)))

    def test_info(self):
        with make_dandi_case(100, old=False, levels=3) as \
                ((url1, volume1), (url2, volume2)):
            ar = DANDIArrayReader([url1, url2])
            info = ar.get_info()
            #
            # Make sure it's json-serializable
            #
            json.dumps(info)
            self.assertEqual(info["data_type"], "uint16")
            self.assertEqual(info["mesh"], "mesh")
            self.assertEqual(info["num_channels"], 1)
            self.assertEqual(info["type"], "image")
            self.assertEqual(len(info["scales"]), 3)
            for scale, level in zip(info["scales"], (1, 2, 4)):
                expected_voxel_size = \
                    [_ * level * 1000 for _ in (2.564, 3.625, 2.564)]
                self.assertSequenceEqual(scale["resolution"],
                                         expected_voxel_size)
                expected_sizes = [_ // level for _ in reversed(ar.shape)]
                self.assertSequenceEqual(scale["size"], expected_sizes)
                self.assertEqual(scale["key"], f"{level}_{level}_{level}")
                self.assertEqual(scale["encoding"], "raw")
                self.assertSequenceEqual(scale["chunk_sizes"][0], (64, 64, 64))
                self.assertSequenceEqual(scale["voxel_offset"], (0, 0, 0))

if __name__ == '__main__':
    unittest.main()
