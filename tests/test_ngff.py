import itertools
import json
import os
import numpy as np
import unittest
import zarr

from precomputed_tif.ngff_stack import NGFFStack
from precomputed_tif.utils import make_case


class TestNGFFStack(unittest.TestCase):
    def test_init(self):
        with make_case(np.uint16, (100, 200, 300), klass=NGFFStack) \
                as (stack, npstack):
            self.assertEqual(stack.x_extent, 300)
            self.assertEqual(stack.y_extent, 200)
            self.assertEqual(stack.z_extent, 100)

    def test_write_info_file(self):
        with make_case(np.uint16, (101, 200, 300), klass=NGFFStack) \
                as (stack, npstack):
            stack.create()
            stack.write_info_file(2)
            scale_md = stack.zgroup.attrs["multiscales"][0]
            for i, dataset_md in enumerate(scale_md["datasets"]):
                self.assertEqual(str(i), dataset_md["path"])

            with open(os.path.join(stack.dest, "info")) as fd:
                info = json.load(fd)
            self.assertEqual(info["data_type"], "uint16")
            self.assertEqual(info["num_channels"], 1)
            scales = info["scales"]
            self.assertEqual(len(scales), 2)
            self.assertSequenceEqual(scales[0]["chunk_sizes"][0], (64, 64, 64))
            self.assertEqual(scales[0]["encoding"], "raw")
            self.assertEqual(scales[0]["key"], "1_1_1")
            self.assertSequenceEqual(scales[0]["resolution"],
                                     (1800, 1800, 2000))
            self.assertSequenceEqual(scales[0]["size"], (300, 200, 101))
            self.assertSequenceEqual(scales[0]["voxel_offset"], (0, 0, 0))
            self.assertSequenceEqual(scales[1]["size"], (150, 100, 51))

    def test_write_level_1(self):
        with make_case(np.uint16, (100, 200, 300), klass=NGFFStack) \
                as (stack, npstack):
            stack.create()
            stack.write_info_file(1)
            stack.write_level_1()
            z_arr = stack.zgroup["0"]
            block_0_64_256 = z_arr[0, 0, :64, 64:128, 256:]
            np.testing.assert_equal(block_0_64_256, npstack[:64, 64:128, 256:])
            for (x0, x1), (y0, y1), (z0, z1) in itertools.product(
                zip(stack.x0(1), stack.x1(1)),
                zip(stack.y0(1), stack.y1(1)),
                zip(stack.z0(1), stack.z1(1))):
                block = z_arr[0, 0, z0:z1, y0:y1, x0:x1]
                np.testing.assert_equal(block, npstack[z0:z1, y0:y1, x0:x1])

    def test_write_level_2(self):
        with make_case(np.uint16, (100, 201, 300), klass=NGFFStack) \
                as (stack, npstack):
            stack.create()
            stack.write_info_file(2)
            stack.write_level_1()
            stack.write_level_n(2)
            dest_lvl2 = os.path.join(stack.dest, "1")
            store = zarr.NestedDirectoryStore(dest_lvl2)
            z_arr = zarr.open(store, "r")
            block = z_arr[0, 0, 0:50, 64:101, 128:150]
            self.assertSequenceEqual(block.shape, (50, 101-64, 150-128))
            s32 = npstack.astype(np.uint32)
            first = (s32[0, 128, 256] + s32[1, 128, 256] +
                     s32[0, 129, 256] + s32[1, 129, 256] +
                     s32[0, 128, 257] + s32[1, 128, 257] +
                     s32[0, 129, 257] + s32[1, 129, 257]) / 8
            self.assertLessEqual(abs(block[0, 0, 0] - first), 1)
            last = (s32[-2, -1, -2] + s32[-1, -1, -2] + s32[-2, -1, -1] +
                    s32[-1, -1, -1]) / 4
            self.assertLessEqual(abs(block[-1, -1, -1]- last), 1)


if __name__ == '__main__':
    unittest.main()
