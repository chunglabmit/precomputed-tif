import glob
import json
import itertools
import numpy as np
import tifffile
import zarr
from numcodecs import Blosc
import os
import tqdm


"""
Default chunk size is (64, 64, 64)
"""

class ZarrStack:

    def __init__(self, glob_expr, dest, compressor=None):
        """

        :param glob_expr: the glob file expression for capturing the files in
        the stack, e.g. "/path/to/img_*.tif*"
        :param dest: the destination folder for zarr arrays
        :param compressor: numcodecs compressor to use on eachj chunk. Default
        is Zstd level 1 with bitshuffle
        """
        self.files = sorted(glob.glob(glob_expr))
        self.z_extent = len(self.files)
        img0 = tifffile.imread(self.files[0])  # need to take this out into main.py to allow zarr precomputed too
        self.y_extent, self.x_extent = img0.shape
        self.dtype = img0.dtype
        self.dest = dest
        if compressor is None:
            self.compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
        else:
            self.compressor = compressor

    def resolution(self, level):
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
        resolution = self.resolution(level)
        return (self.x_extent // resolution + 63) // 64

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
        resolution = self.resolution(level)
        return (self.y_extent // resolution + 63) // 64

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
        resolution = self.resolution(level)
        return (self.z_extent // resolution + 63) // 64

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

    def write_level_1(self, silent=False):
        """Write the first mipmap level, loading from tiff planes"""
        dest = os.path.join(self.dest, "1_1_1")
        store = zarr.NestedDirectoryStore(dest)

        z_arr_1 = zarr.open(store,
                            mode='w',
                            chunks=(64, 64, 64),
                            dtype=self.dtype,
                            shape=(self.z_extent, self.y_extent, self.x_extent),
                            compression=self.compressor)

        z0 = self.z0(1)
        z1 = self.z1(1)
        y0 = self.y0(1)
        y1 = self.y1(1)
        x0 = self.x0(1)
        x1 = self.x1(1)

        for z0a, z1a in tqdm.tqdm(zip(z0, z1), total=len(z0), disable=silent):
            img = np.zeros((64, y1[-1], x1[-1]), self.dtype)
            for z in range(z0a, z1a):
                img[z-z0a] = tifffile.imread(self.files[z])
            z_arr_1[z0a:z1a] = img

    def write_level_n(self, level, silent=False):
        src_resolution = self.resolution(level - 1)
        dest_resolution = self.resolution(level)

        src = os.path.join(
            self.dest,
            "%d_%d_%d" % (src_resolution, src_resolution, src_resolution))
        dest = os.path.join(
            self.dest,
            "%d_%d_%d" % (dest_resolution, dest_resolution, dest_resolution))

        src_store = zarr.NestedDirectoryStore(src)
        src_zarr = zarr.open(src_store, mode='r')

        dest_store = zarr.NestedDirectoryStore(dest)
        dest_zarr = zarr.open(dest_store,
                              mode='w',
                              chunks=(64, 64, 64),
                              dtype=self.dtype,
                              shape=(self.z1(level)[-1],
                                     self.y1(level)[-1],
                                     self.x1(level)[-1]),
                              compression=self.compressor)

        z0s = self.z0(level - 1)  # source block coordinates
        z1s = self.z1(level - 1)
        y0s = self.y0(level - 1)
        y1s = self.y1(level - 1)
        x0s = self.x0(level - 1)
        x1s = self.x1(level - 1)
        z0d = self.z0(level)  # dest block coordinates
        z1d = self.z1(level)
        y0d = self.y0(level)
        y1d = self.y1(level)
        x0d = self.x0(level)
        x1d = self.x1(level)

        for xidx, yidx, zidx in tqdm.tqdm(list(itertools.product(
                range(self.n_x(level)),
                range(self.n_y(level)),
                range(self.n_z(level)))),
            disable=silent):  # looping over destination block indicies (fewer blocks than source)
            block = np.zeros((z1d[zidx] - z0d[zidx],
                              y1d[yidx] - y0d[yidx],
                              x1d[xidx] - x0d[xidx]), np.uint64)
            hits = np.zeros((z1d[zidx] - z0d[zidx],
                             y1d[yidx] - y0d[yidx],
                             x1d[xidx] - x0d[xidx]), np.uint64)
            for xsi1, ysi1, zsi1 in itertools.product((0, 1), (0, 1), (0, 1)):  # looping over source blocks for this destination
                xsi = xsi1 + xidx * 2
                if xsi == self.n_x(level-1):  # Check for any source blocks that are out-of-bounds
                    continue
                ysi = ysi1 + yidx * 2
                if ysi == self.n_y(level-1):
                    continue
                zsi = zsi1 + zidx * 2
                if zsi == self.n_z(level-1):
                    continue

                src_block = src_zarr[z0s[zsi]:z1s[zsi],
                                     y0s[ysi]:y1s[ysi],
                                     x0s[xsi]:x1s[xsi]]

                for offx, offy, offz in \
                        itertools.product((0, 1), (0, 1), (0,1)):
                    dsblock = src_block[offz::2, offy::2, offx::2]
                    block[zsi1*32:zsi1*32 + dsblock.shape[0],
                          ysi1*32:ysi1*32 + dsblock.shape[1],
                          xsi1*32:xsi1*32 + dsblock.shape[2]] += \
                        dsblock.astype(block.dtype)  # 32 is half-block size of source
                    hits[zsi1*32:zsi1*32 + dsblock.shape[0],
                         ysi1*32:ysi1*32 + dsblock.shape[1],
                         xsi1*32:xsi1*32 + dsblock.shape[2]] += 1
            block[hits > 0] = block[hits > 0] // hits[hits > 0]

            dest_zarr[z0d[zidx]:z1d[zidx],
                      y0d[yidx]:y1d[yidx],
                      x0d[xidx]:x1d[xidx]] = block
