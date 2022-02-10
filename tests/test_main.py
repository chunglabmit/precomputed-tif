import numpy as np
import os
import unittest
import zarr
from precomputed_tif.utils import make_case
from precomputed_tif.main import main
from blockfs.directory import Directory
from precomputed_tif.blockfs_stack import BlockfsStack

class TestMain(unittest.TestCase):
    def test_tiff(self):
        with make_case(np.uint16, (256, 256, 256),
                       return_path=True) as (glob_expr, dest, volume):
            main(["--source", glob_expr,
                  "--dest", dest,
                  "--levels", "2",
                  "--format", "tiff"])
            main(["--source", glob_expr,
                  "--dest", dest,
                  "--levels", "2",
                  "--format", "tiff",
                  "--chunk-size", "128,128,128"])

    def test_zarr(self):
        with make_case(np.uint16, (128, 128, 128),
                       return_path=True) as (glob_expr, dest, volume):
            main(["--source", glob_expr,
                  "--dest", dest,
                  "--levels", "2",
                  "--format", "zarr"])
            store = zarr.NestedDirectoryStore(os.path.join(dest, "1_1_1"))
            z_arr = zarr.open(store, "r")
            self.assertSequenceEqual(z_arr.shape, (128, 128, 128))

    def test_blockfs(self):
        with make_case(np.uint16, (256, 256, 256), return_path=True) as \
                (glob_expr, dest, volume):
            main(["--source", glob_expr,
                  "--dest", dest,
                  "--levels", "2",
                  "--format", "blockfs"])
            directory = Directory.open(os.path.join(
                dest, "1_1_1", BlockfsStack.DIRECTORY_FILENAME))
            block = directory.read_block(0, 0, 0)
            np.testing.assert_array_equal(block, volume[:64, :64, :64])
            main(["--source", glob_expr,
                  "--dest", dest,
                  "--levels", "2",
                  "--format", "blockfs",
                  "--chunk-size", "128,128,128"])
            directory = Directory.open(os.path.join(
                dest, "1_1_1", BlockfsStack.DIRECTORY_FILENAME))
            block = directory.read_block(0, 0, 0)
            np.testing.assert_array_equal(block, volume[:128, :128, :128])


    def test_n_cores(self):
        with make_case(np.uint16, (10, 20, 30),
                       return_path=True) as (glob_expr, dest, volume):
            main(["--source", glob_expr,
                  "--dest", dest,
                  "--levels", "2",
                  "--format", "tiff",
                  "--n-cores", "1"])


if __name__ == '__main__':
    unittest.main()
