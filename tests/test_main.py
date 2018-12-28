import numpy as np
import unittest
from precomputed_tif.utils import make_case
from precomputed_tif.main import main

class TestMain(unittest.TestCase):
    def test_tiff(self):
        with make_case(np.uint16, (128, 128, 128),
                       return_path=True) as (glob_expr, dest, volume):
            main(["--source", glob_expr,
                  "--dest", dest,
                  "--levels", "2",
                  "--format", "tiff"])

    def test_zarr(self):
        with make_case(np.uint16, (128, 128, 128),
                       return_path=True) as (glob_expr, dest, volume):
            main(["--source", glob_expr,
                  "--dest", dest,
                  "--levels", "2",
                  "--format", "zarr"])

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
