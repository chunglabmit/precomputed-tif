import numpy as np
import os
import unittest
from precomputed_tif.blockfs_stack import BlockfsStack
from precomputed_tif.utils import make_case
from blockfs import Directory
import logging
logging.basicConfig(level=logging.DEBUG)
class TestBlockfsStack(unittest.TestCase):
    def test_write_oddly_shaped(self):
        with make_case(np.uint16, (100, 100, 100), return_path=True)\
            as (glob_expr, dest, stack):
            bfs = BlockfsStack(glob_expr, dest)
            bfs.write_level_1(silent=True, n_cores=1)
            directory_filename =\
                os.path.join(dest, "1_1_1", BlockfsStack.DIRECTORY_FILENAME)
            directory =  Directory.open(directory_filename)
            np.testing.assert_array_equal(stack[:64, :64, :64],
                                          directory.read_block(0, 0, 0))
            np.testing.assert_array_equal(stack[64:, 64:, 64:],
                                          directory.read_block(64, 64, 64))

if __name__ == '__main__':
    unittest.main()
