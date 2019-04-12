import glob
import json
import itertools
import numpy as np
import tifffile
import os
import tqdm
import multiprocessing


class StackBase:
    def __init__(self, glob_expr, dest):
        """

        :param glob_expr: the glob file expression for capturing the files in
        the stack, e.g. "/path/to/img_*.tif*"
        :param dest: the destination root directory for the precomputed files
        """
        self.files = sorted(glob.glob(glob_expr))
        self.z_extent = len(self.files)
        img0 = tifffile.imread(self.files[0])
        self.y_extent, self.x_extent = img0.shape
        self.dtype = img0.dtype
        self.dest = dest

    @staticmethod
    def resolution(level):
        """The pixel resolution at a given level

        :param level: 1 to N, the mipmap level
        :returns: the number of pixels at the base level per pixel at this
        level
        """
        return 2 ** (level - 1)

    def n_x(self, level):
        """The number of blocks in the X direction at the given level

        :param level: mipmap level, starting at 1
        :return: # of blocks in the X direction
        """
        return (self.x_extent // (2 ** (level - 1)) + 63) // 64

    def x0(self, level):
        """The starting X coordinates at a particular level

        :param level: 1 to N
        :return: an array of starting X coordinates
        """
        resolution = self.resolution(level)
        return np.arange(0, (self.x_extent + resolution - 1) // resolution, 64)

    def x1(self, level):
        """The ending X coordinates at a particular level

        :param level: the mipmap level (1 to N)
        :return: an array of ending X coordinates
        """
        resolution = self.resolution(level)
        x1 = self.x0(level) + 64
        x1[-1] = (self.x_extent + resolution - 1) // resolution
        return x1

    def n_y(self, level):
        """The number of blocks in the Y direction at the given level

        :param level: mipmap level, starting at 1
        :return: # of blocks in the Y direction
        """
        return (self.y_extent // (2 ** (level - 1)) + 63) // 64

    def y0(self, level):
        """The starting Y coordinates at a particular level

        :param level: 1 to N
        :return: an array of starting Y coordinates
        """
        resolution = self.resolution(level)
        return np.arange(0, (self.y_extent + resolution - 1) // resolution, 64)

    def y1(self, level):
        """The ending Y coordinates at a particular level

        :param level: the mipmap level (1 to N)
        :return: an array of ending Y coordinates
        """
        resolution = self.resolution(level)
        y1 = self.y0(level) + 64
        y1[-1] = (self.y_extent + resolution - 1) // resolution
        return y1

    def n_z(self, level):
        """The number of blocks in the Z direction at the given level

        :param level: mipmap level, starting at 1
        :return: # of blocks in the Z direction
        """
        return (self.z_extent // (2 ** (level - 1)) + 63) // 64

    def z0(self, level):
        """The starting Z coordinates at a particular level

        :param level: 1 to N
        :return: an array of starting Z coordinates
        """
        resolution = self.resolution(level)
        return np.arange(0, (self.z_extent + resolution - 1) // resolution, 64)

    def z1(self, level):
        """The ending Z coordinates at a particular level

        :param level: the mipmap level (1 to N)
        :return: an array of ending Z coordinates
        """
        resolution = self.resolution(level)
        z1 = self.z0(level) + 64
        z1[-1] = (self.z_extent + resolution - 1) // resolution
        return z1

    def write_info_file(self, n_levels):
        """Write the precomputed info file that defines the volume

        :param n_levels: the number of levels to be written
        """
        if not os.path.exists(self.dest):
            os.mkdir(self.dest)
        d = dict(data_type = self.dtype.name,
                 mesh="mesh",
                 num_channels=1,
                 type="image")
        scales = []
        z_extent = self.z_extent
        y_extent = self.y_extent
        x_extent = self.x_extent
        for level in range(1, n_levels + 1):
            resolution = self.resolution(level)
            scales.append(
                dict(chunk_sizes=[[64, 64, 64]],
                     encoding="raw",
                     key="%d_%d_%d" % (resolution, resolution, resolution),
                     resolution=[resolution, resolution, resolution],
                     size=[x_extent, y_extent, z_extent],
                     voxel_offset=[0, 0, 0]))
            z_extent = (z_extent + 1) // 2
            y_extent = (y_extent + 1) // 2
            x_extent = (x_extent + 1) // 2
        d["scales"] = scales
        with open(os.path.join(self.dest, "info"), "w") as fd:
            json.dump(d, fd, indent=2, sort_keys=True)


class Stack(StackBase):

    def __init__(self, glob_expr, dest):
        super(Stack, self).__init__(glob_expr, dest)

    def fname(self, level, x0, x1, y0, y1, z0, z1):
        return Stack.sfname(self.dest, level, x0, x1, y0, y1, z0, z1)

    @staticmethod
    def sfname(dest, level, x0, x1, y0, y1, z0, z1):
        """The file name of the block with these coordinates

        :param level: the mipmap level of the block
        :param x0: starting X of the block
        :param x1: ending X of the block
        :param y0: starting Y of the block
        :param y1: ending Y of the block
        :param z0: starting Z of the block
        :param z1: ending Z of the block
        :return:
        """
        resolution = Stack.resolution(level)
        return os.path.join(dest,
                     "%d_%d_%d" % (resolution, resolution, resolution),
                     "%d-%d_%d-%d_%d-%d.tiff" % (x0, x1, y0, y1, z0, z1))

    def write_level_1(self, silent=False,
                      n_cores=min(os.cpu_count(), 4)):
        """Write the first mipmap level, loading from tiff planes"""
        dest = os.path.join(self.dest, "1_1_1")
        if not os.path.exists(dest):
            os.mkdir(dest)
        z0 = self.z0(1)
        z1 = self.z1(1)
        y0 = self.y0(1)
        y1 = self.y1(1)
        x0 = self.x0(1)
        x1 = self.x1(1)
        x_extent = self.x_extent
        y_extent = self.y_extent
        dest = self.dest
        dtype = self.dtype
        with multiprocessing.Pool(n_cores) as pool:
            futures = []
            for z0a, z1a in zip(z0, z1):
                files = self.files[z0a:z1a]
                futures.append(pool.apply_async(
                    Stack.write_one_level_1,
                    (dest, dtype, files, x_extent, y_extent,
                     x0, x1, y0, y1, z0a, z1a)))
            for future in tqdm.tqdm(futures, disable=silent):
                future.get()

    @staticmethod
    def write_one_level_1(dest, dtype, files, x_extent, y_extent,
                          x0, x1, y0, y1, z0a, z1a):
        img = np.zeros((64, y0[-1] + 64, x0[-1] + 64), dtype)
        for z, file in zip(range(z0a, z1a), files):
            img[z - z0a, :y_extent, :x_extent] = \
                tifffile.imread(file)
        for (x0a, x1a), (y0a, y1a) in itertools.product(
                zip(x0, x1), zip(y0, y1)):
            path = Stack.sfname(dest, 1, x0a, x1a, y0a, y1a, z0a, z1a)
            tifffile.imsave(path, img[:z1a - z0a, y0a:y1a, x0a:x1a],
                            compress=4)

    def write_level_n(self, level, silent=False,
                      n_cores = min(os.cpu_count(), 12)):
        src_resolution = self.resolution(level - 1)
        dest_resolution = self.resolution(level)
        dest = os.path.join(
            self.dest,
            "%d_%d_%d" % (dest_resolution, dest_resolution, dest_resolution))
        if not os.path.exists(dest):
            os.mkdir(dest)
        z0s = self.z0(level - 1)
        z1s = self.z1(level - 1)
        y0s = self.y0(level - 1)
        y1s = self.y1(level - 1)
        x0s = self.x0(level - 1)
        x1s = self.x1(level - 1)
        z0d = self.z0(level)
        z1d = self.z1(level)
        y0d = self.y0(level)
        y1d = self.y1(level)
        x0d = self.x0(level)
        x1d = self.x1(level)
        dest = self.dest
        dtype = self.dtype
        xsi_max = self.n_x(level - 1)
        ysi_max = self.n_y(level - 1)
        zsi_max = self.n_z(level - 1)
        with multiprocessing.Pool(n_cores) as pool:
            futures = []
            for xidx, yidx, zidx in itertools.product(
                    range(self.n_x(level)),
                    range(self.n_y(level)),
                    range(self.n_z(level))):
                futures.append(pool.apply_async(
                    Stack.write_one_level_n,
                    (dest, dtype, level, x0d, x0s, x1d, x1s, xidx, xsi_max,
                     y0d, y0s, y1d, y1s, yidx, ysi_max,
                     z0d, z0s, z1d, z1s, zidx, zsi_max)))
            for future in tqdm.tqdm(futures):
                future.get()

    @staticmethod
    def write_one_level_n(dest, dtype, level, x0d, x0s, x1d, x1s, xidx, xsi_max,
                          y0d, y0s, y1d, y1s, yidx, ysi_max,
                          z0d, z0s, z1d, z1s, zidx, zsi_max):
        block = np.zeros((z1d[zidx] - z0d[zidx],
                          y1d[yidx] - y0d[yidx],
                          x1d[xidx] - x0d[xidx]), np.uint64)
        hits = np.zeros((z1d[zidx] - z0d[zidx],
                         y1d[yidx] - y0d[yidx],
                         x1d[xidx] - x0d[xidx]), np.uint64)
        for xsi1, ysi1, zsi1 in itertools.product((0, 1), (0, 1), (0, 1)):
            xsi = xsi1 + xidx * 2
            if xsi == xsi_max:
                continue
            ysi = ysi1 + yidx * 2
            if ysi == ysi_max:
                continue
            zsi = zsi1 + zidx * 2
            if zsi == zsi_max:
                continue
            src_path = Stack.sfname(
                dest, level - 1, x0s[xsi], x1s[xsi], y0s[ysi], y1s[ysi],
                z0s[zsi], z1s[zsi])
            src_block = tifffile.imread(src_path)
            for offx, offy, offz in \
                    itertools.product((0, 1), (0, 1), (0, 1)):
                dsblock = src_block[offz::2, offy::2, offx::2]
                block[zsi1 * 32:zsi1 * 32 + dsblock.shape[0],
                ysi1 * 32:ysi1 * 32 + dsblock.shape[1],
                xsi1 * 32:xsi1 * 32 + dsblock.shape[2]] += \
                    dsblock.astype(block.dtype)
                hits[zsi1 * 32:zsi1 * 32 + dsblock.shape[0],
                ysi1 * 32:ysi1 * 32 + dsblock.shape[1],
                xsi1 * 32:xsi1 * 32 + dsblock.shape[2]] += 1
        block[hits > 0] = block[hits > 0] // hits[hits > 0]
        dest_path = Stack.sfname(
            dest, level, x0d[xidx], x1d[xidx], y0d[yidx], y1d[yidx],
            z0d[zidx], z1d[zidx])
        tifffile.imsave(dest_path,
                        block[:,
                        :y1d[yidx] - y0d[yidx],
                        :x1d[xidx] - x0d[xidx]].astype(dtype),
                        compress=4)
