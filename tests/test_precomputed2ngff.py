import pathlib
import shutil
import tempfile
import unittest

import numpy as np
import zarr
from precomputed_tif.client import ArrayReader

from precomputed_tif.utils import make_case
from precomputed_tif.precomputed2ngff import main as precomputed2ngff
from precomputed_tif.main import main as tiff2precomputed


class TestPrecomputed2NGFF(unittest.TestCase):
    def test_one_level(self):
        with make_case(np.uint16, (100, 200, 300),
                       return_path=True) as (glob_expr, dest, volume):
            tiff2precomputed([
                  "--source", glob_expr,
                  "--dest", dest,
                  "--levels", "1",
                  "--format", "blockfs"])
            dest_path = pathlib.Path(dest)
            target = tempfile.mkdtemp(".ngff")
            try:
                precomputed2ngff([
                    "--input", dest_path.as_uri(),
                    "--input-format", "blockfs",
                    "--output", target
                ])
                storage = zarr.NestedDirectoryStore(target)
                group = zarr.group(storage, overwrite=False)
                array = group["0"]
                self.assertSequenceEqual(array.shape, (1, 1, 100, 200, 300))
                np.testing.assert_array_equal(array[0, 0], volume)
            finally:
                shutil.rmtree(target)

    def test_two_levels(self):
        with make_case(np.uint16, (100, 200, 300),
                       return_path=True) as (glob_expr, dest, volume):
            tiff2precomputed([
                  "--source", glob_expr,
                  "--dest", dest,
                  "--levels", "2",
                  "--format", "blockfs"])
            dest_uri = pathlib.Path(dest).as_uri()
            target = tempfile.mkdtemp(".ngff")
            try:
                precomputed2ngff([
                    "--input", dest_uri,
                    "--input-format", "blockfs",
                    "--output", target
                ])
                storage = zarr.NestedDirectoryStore(target)
                group = zarr.group(storage, overwrite=False)
                array = group["0"]
                self.assertSequenceEqual(array.shape, (1, 1, 100, 200, 300))
                np.testing.assert_array_equal(array[0, 0], volume)
                array = group["1"]
                self.assertSequenceEqual(array.shape, (1, 1, 50, 100, 150))
                ar = ArrayReader(dest_uri, format="blockfs", level=2)
                np.testing.assert_array_equal(ar[:, :, :], array[0, 0])
            finally:
                shutil.rmtree(target)

if __name__ == '__main__':
    unittest.main()
