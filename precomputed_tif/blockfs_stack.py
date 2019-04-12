import glob
from numcodecs import Blosc
import multiprocessing
import numpy as np
import os
import threading
import json
import tifffile
import tqdm
from blockfs import Directory, Compression
import uuid
import itertools

from .stack import StackBase

directories = {}

class BlockfsStack(StackBase):

    DIRECTORY_FILENAME="precomputed.blockfs"

    def __init__(self, glob_expr, dest):
        super(BlockfsStack, self).__init__(glob_expr, dest)

    def fname(self, level):
        return Stack.resolution(level) + ".blockfs"

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
        z_extent = self.z_extent
        dtype = self.dtype
        directory_filename = os.path.join(dest, BlockfsStack.DIRECTORY_FILENAME)
        block_files = [directory_filename + ".%d" % _ for _ in range(n_cores)]

        directory = Directory(x_extent, y_extent, z_extent, dtype,
                              directory_filename=directory_filename,
                              block_filenames=block_files)
        directory.create()
        directory.start_writer_processes()
        directory_id = uuid.uuid4()
        directories[directory_id] = directory

        with multiprocessing.Pool(n_cores) as pool:
            futures = []
            for z0a, z1a in zip(z0, z1):
                files = self.files[z0a:z1a]
                futures.append(pool.apply_async(
                    BlockfsStack.write_one_level_1,
                    (directory_id, files,
                     x0, x1, y0, y1, z0a, z1a)))
            for future in tqdm.tqdm(futures, disable=silent):
                future.get()
        try:
            acc = 0
            for bw in directory.writers:
                q = bw.q_in
                acc += q.qsize()
            print("Waiting for %d blocks to be written" % acc)
        except:
            pass
        directory.close()

    @staticmethod
    def write_one_level_1(directory_id, files,
                          x0, x1, y0, y1, z0a, z1a):
        directory:Directory = directories[directory_id]
        x_extent = directory.x_extent
        y_extent = directory.y_extent
        dtype = directory.dtype
        img = np.zeros((64, y0[-1] + 64, x0[-1] + 64), dtype)
        for z, file in zip(range(z0a, z1a), files):
            img[z - z0a, :y_extent, :x_extent] = \
                tifffile.imread(file)
        for (x0a, x1a), (y0a, y1a) in itertools.product(
                zip(x0, x1), zip(y0, y1)):
            block = img[:z1a - z0a, y0a:y1a, x0a:x1a]
            directory.write_block(block, x0a, y0a, z0a)

    def write_level_n(self, level, silent=False,
                      n_cores = min(os.cpu_count(), 12)):
        src_resolution = self.resolution(level - 1)
        dest_resolution = self.resolution(level)
        dest = os.path.join(
            self.dest,
            "%d_%d_%d" % (dest_resolution, dest_resolution, dest_resolution))
        if not os.path.exists(dest):
            os.mkdir(dest)
        src = os.path.join(
            self.dest,
            "%d_%d_%d" % (src_resolution, src_resolution, src_resolution))
        src_directory_filename = \
            os.path.join(src, BlockfsStack.DIRECTORY_FILENAME)
        src_directory = Directory.open(src_directory_filename)
        src_directory_id = uuid.uuid4()
        directories[src_directory_id] = src_directory

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
        dest_directory_filename = \
            os.path.join(dest, BlockfsStack.DIRECTORY_FILENAME)
        dest_block_filenames = \
            [dest_directory_filename + ".%d" % _ for _ in range(n_cores)]
        dest_directory = Directory(x1d[-1] - x0d[0],
                                   y1d[-1] - y0d[0],
                                   z1d[-1] - z0d[0],
                                   self.dtype,
                                   dest_directory_filename,
                                   block_filenames=dest_block_filenames)
        dest_directory.create()
        dest_directory_id = uuid.uuid4()
        directories[dest_directory_id] = dest_directory
        dest_directory.start_writer_processes()
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
                    BlockfsStack.write_one_level_n,
                    (src_directory_id, dest_directory_id,
                     x0d, x0s, x1d, x1s, xidx, xsi_max,
                     y0d, y0s, y1d, y1s, yidx, ysi_max,
                     z0d, z0s, z1d, z1s, zidx, zsi_max)))
            for future in tqdm.tqdm(futures):
                future.get()
        dest_directory.close()

    @staticmethod
    def write_one_level_n(src_directory_id, dest_directory_id,
                          x0d, x0s, x1d, x1s, xidx, xsi_max,
                          y0d, y0s, y1d, y1s, yidx, ysi_max,
                          z0d, z0s, z1d, z1s, zidx, zsi_max):
        src_directory:Directory = directories[src_directory_id]
        dest_directory:Directory = directories[dest_directory_id]
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
            src_block = src_directory.read_block(x0s[xsi], y0s[ysi], z0s[zsi])
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
        dest_directory.write_block(block, x0d[xidx], y0d[yidx], z0d[zidx])