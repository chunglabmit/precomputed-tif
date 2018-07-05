import unittest
from precomputed_tif.stack import Stack
import tempfile
import numpy as np
import tifffile
import contextlib
import itertools
import json
import os
import shutil

class TestStack(unittest.TestCase):

    @contextlib.contextmanager
    def make_case(self, dtype, shape):
        imax = np.iinfo(dtype).max
        npstack = np.random.RandomState(1234).randint(
            0, imax, size=shape).astype(dtype)
        tempdir = tempfile.mkdtemp()
        src = os.path.join(tempdir, "src")
        os.mkdir(src)
        dest = os.path.join(tempdir, "dest")
        os.mkdir(dest)
        try:
            for z in range(shape[0]):
                tifffile.imsave(os.path.join(src, "img_%04d.tiff" % z),
                                npstack[z])
                stack = Stack(os.path.join(src, "img_*.tiff"), dest)
            yield stack, npstack
        finally:
            shutil.rmtree(tempdir)

    def test_init(self):
        with self.make_case(np.uint16, (100, 200, 300)) as (stack, npstack):
            self.assertEqual(stack.x_extent, 300)
            self.assertEqual(stack.y_extent, 200)
            self.assertEqual(stack.z_extent, 100)

    def test_coordinates(self):
        with self.make_case(np.uint16, (100, 200, 300)) as (stack, npstack):
            self.assertEqual(stack.n_x(1), 300 // 64 + 1)
            self.assertEqual(stack.n_x(2), 3)
            self.assertEqual(stack.n_x(3), 2)
            self.assertEqual(stack.n_x(4), 1)
            self.assertEqual(stack.n_x(5), 1)
            self.assertEqual(stack.n_y(1), 4)
            self.assertEqual(stack.n_y(2), 2)
            self.assertEqual(stack.n_z(1), 2)
            self.assertEqual(stack.n_z(2), 1)
            self.assertSequenceEqual(stack.z0(1).tolist(), (0, 64))
            self.assertSequenceEqual(stack.z1(1).tolist(), (64, 100))
            self.assertSequenceEqual(stack.y0(1).tolist(), (0, 64, 128, 192))
            self.assertSequenceEqual(stack.y1(1).tolist(), (64, 128, 192, 200))
            self.assertSequenceEqual(stack.x0(2).tolist(), (0, 64, 128))
            self.assertSequenceEqual(stack.x1(2).tolist(), (64, 128, 150))

    def test_write_info_file(self):
        with self.make_case(np.uint16, (101, 200, 300)) as (stack, npstack):
            stack.write_info_file(2)
            with open(os.path.join(stack.dest, "info")) as fd:
                info = json.load(fd)
            self.assertEqual(info["data_type"], "uint16")
            self.assertEqual(info["num_channels"], 1)
            scales = info["scales"]
            self.assertEqual(len(scales), 2)
            self.assertSequenceEqual(scales[0]["chunk_sizes"], (64, 64, 64))
            self.assertEqual(scales[0]["encoding"], "raw")
            self.assertEqual(scales[0]["key"], "1_1_1")
            self.assertSequenceEqual(scales[0]["resolution"], (1, 1, 1))
            self.assertSequenceEqual(scales[0]["size"], (101, 200, 300))
            self.assertSequenceEqual(scales[0]["voxel_offset"], (0, 0, 0))
            self.assertSequenceEqual(scales[1]["size"], (51, 100, 150))

    def test_write_level_1(self):
        with self.make_case(np.uint16, (100, 200, 300)) as (stack, npstack):
            stack.write_level_1()
            block_0_64_256 = tifffile.imread(
                os.path.join(stack.dest, "1_1_1", "256-300_64-128_0-64.tiff"))
            np.testing.assert_equal(block_0_64_256, npstack[:64, 64:128, 256:])
            for (x0, x1), (y0, y1), (z0, z1) in itertools.product(
                zip(stack.x0(1), stack.x1(1)),
                zip(stack.y0(1), stack.y1(1)),
                zip(stack.z0(1), stack.z1(1))):
                path = stack.fname(1, x0, x1, y0, y1, z0, z1)
                block = tifffile.imread(path)
                np.testing.assert_equal(block, npstack[z0:z1, y0:y1, x0:x1])

    def test_write_level_2(self):
        with self.make_case(np.uint16, (100, 201, 300)) as (stack, npstack):
            stack.write_level_1()
            stack.write_level_n(2)
            block = tifffile.imread(
                os.path.join(stack.dest, "2_2_2", "128-150_64-101_0-50.tiff"))
            self.assertSequenceEqual(block.shape, (50, 101-64, 150-128))
            s32 = npstack.astype(np.uint32)
            first = (s32[0, 128, 256] + s32[1, 128, 256] +
                     s32[0, 129, 256] + s32[1, 129, 256] +
                     s32[0, 128, 257] + s32[1, 128, 257] +
                     s32[0, 129, 257] + s32[1, 129, 257]) // 8
            self.assertEqual(block[0, 0, 0], first)
            last = (s32[-2, -1, -2] + s32[-1, -1, -2] + s32[-2, -1, -1] +
                    s32[-1, -1, -1]) // 4
            self.assertEqual(block[-1, -1, -1], last)


if __name__ == '__main__':
    unittest.main()
