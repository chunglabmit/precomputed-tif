import json
import unittest
import unittest.mock
import urllib.request
import numpy as np
import precomputed_tif.client
from precomputed_tif.client import read_chunk, clear_cache

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


if __name__ == '__main__':
    unittest.main()