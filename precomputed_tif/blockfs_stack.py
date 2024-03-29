import logging
from mp_shared_memory import SharedMemory
import multiprocessing
import numpy as np
from scipy import ndimage
import os
import tifffile
import tqdm
from blockfs import Directory, Compression
import uuid
import itertools
from PIL import Image

from .stack import StackBase, PType

logger = logging.getLogger("precomputed_tif.blockfs_stack")
directories = {}


class BlockfsStack(StackBase):

    DIRECTORY_FILENAME="precomputed.blockfs"

    def __init__(self, glob_expr, dest, ptype=PType.IMAGE,
                 chunk_size=(64, 64, 64)):
        super(BlockfsStack, self).__init__(glob_expr, dest, ptype=ptype,
                                           chunk_size=chunk_size)
        if self.dtype == bool: self.dtype = np.dtype(np.uint16)

    def fname(self, level):
        return StackBase.resolution(level) + ".blockfs"

    def write_level_1(self, silent=False,
                      n_cores=min(os.cpu_count(), 4)):
        """Write the first mipmap level, loading from tiff planes"""
        z0 = self.z0(1)
        z1 = self.z1(1)
        y0 = self.y0(1)
        y1 = self.y1(1)
        x0 = self.x0(1)
        x1 = self.x1(1)

        directory = self.make_l1_directory(n_cores)
        directory.create()
        directory.start_writer_processes()
        directory_id = uuid.uuid4()
        directories[directory_id] = directory

        with multiprocessing.Pool(n_cores) as pool:
            for z0a, z1a in tqdm.tqdm(list(zip(z0, z1))):
                files = self.files[z0a:z1a]
                BlockfsStack.write_one_level_1(
                    pool, directory_id, files,
                     x0, x1, y0, y1, z0a, z1a)
            try:
                acc = 0
                for bw in directory.writers:
                    q = bw.q_in
                    acc += q.qsize()
                logger.info("Waiting for %d blocks to be written" % acc)
            except:
                pass
            directory.close()

    def make_l1_directory(self, n_cores):
        """
        Make the level 1 directory object for this stack

        :param n_cores: # of writers to use
        :return: blockfs directory structure (needs directory.create(),
        directory.start_writer_processes() and directory.close()
        """
        dest = os.path.join(self.dest, "1_1_1")
        if not os.path.exists(dest):
            os.mkdir(dest)
        x_extent = self.x_extent
        y_extent = self.y_extent
        z_extent = self.z_extent
        dtype = self.dtype
        directory_filename = \
            os.path.abspath(os.path.join(dest, BlockfsStack.DIRECTORY_FILENAME))
        block_files = [directory_filename + ".%d" % _ for _ in range(n_cores)]
        directory = Directory(x_extent, y_extent, z_extent, dtype,
                              x_block_size=self.cx(),
                              y_block_size=self.cy(),
                              z_block_size=self.cz(),
                              directory_filename=directory_filename,
                              block_filenames=block_files)
        return directory

    @staticmethod
    def read_tiff(shm, z, path):
        with shm.txn() as m:
            if '.jpeg' in path or '.jpg' in path:
                jarray = np.asarray(Image.open(path))
                jarray[jarray < 20] = 0
                m[z] = jarray
            else:
                arr = tifffile.imread(path)
                if arr.dtype == bool: 
                    arr = (arr.astype(np.uint16)*100)
                m[z] = arr

    @staticmethod
    def write_one_level_1(pool, directory_id, files,
                          x0, x1, y0, y1, z0a, z1a):
        directory = directories[directory_id]
        x_extent = directory.x_extent
        y_extent = directory.y_extent
        dtype = directory.dtype
        shm = SharedMemory((z1a - z0a, y_extent, x_extent),
                           dtype)
        img = np.zeros((64, y0[-1] + 64, x0[-1] + 64), dtype)
        futures = []
        for z, file in zip(range(z0a, z1a), files):
            futures.append(pool.apply_async(BlockfsStack.read_tiff,
                                            (shm, z-z0a, file)))
        for future in futures:
            future.get()
        with shm.txn() as img:
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
            os.path.abspath(os.path.join(dest, BlockfsStack.DIRECTORY_FILENAME))
        dest_block_filenames = \
            [dest_directory_filename + ".%d" % _ for _ in range(n_cores)]
        dest_directory = Directory(x1d[-1] - x0d[0],
                                   y1d[-1] - y0d[0],
                                   z1d[-1] - z0d[0],
                                   self.dtype,
                                   dest_directory_filename,
                                   x_block_size=self.cx(),
                                   y_block_size=self.cy(),
                                   z_block_size=self.cz(),
                                   block_filenames=dest_block_filenames)
        dest_directory.create()
        dest_directory_id = uuid.uuid4()
        directories[dest_directory_id] = dest_directory
        dest_directory.start_writer_processes()
        xsi_max = self.n_x(level - 1)
        ysi_max = self.n_y(level - 1)
        zsi_max = self.n_z(level - 1)
        if self.ptype == PType.IMAGE:
            fn = BlockfsStack.write_one_level_n
        else:
            fn = BlockfsStack.write_stack_level_n
        with multiprocessing.Pool(n_cores) as pool:
            try:
                futures = []
                for xidx, yidx, zidx in itertools.product(
                        range(self.n_x(level)),
                        range(self.n_y(level)),
                        range(self.n_z(level))):
                    futures.append(pool.apply_async(
                        fn,
                        (src_directory_id, dest_directory_id,
                         x0d, x0s, x1d, x1s, xidx, xsi_max,
                         y0d, y0s, y1d, y1s, yidx, ysi_max,
                         z0d, z0s, z1d, z1s, zidx, zsi_max)))
                for future in tqdm.tqdm(futures):
                    future.get()
            finally:
                dest_directory.close()

    @staticmethod
    def write_one_level_n(src_directory_id, dest_directory_id,
                          x0d, x0s, x1d, x1s, xidx, xsi_max,
                          y0d, y0s, y1d, y1s, yidx, ysi_max,
                          z0d, z0s, z1d, z1s, zidx, zsi_max):
        logger.debug(f"Writing {x0s}:{x1s},{y0s}:{y1s},{z0s}:{z1s} to"
                     f"{x0d}:{x1d},{y0d}:{y1d},{z0d}:{z1d}")
        src_directory = directories[src_directory_id]
        dest_directory = directories[dest_directory_id]
        hx = src_directory.x_block_size // 2
        hy = src_directory.y_block_size // 2
        hz = src_directory.z_block_size // 2
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
                block[zsi1 * hz:zsi1 * hz + dsblock.shape[0],
                ysi1 * hy:ysi1 * hy + dsblock.shape[1],
                xsi1 * hx:xsi1 * hx + dsblock.shape[2]] += \
                    dsblock.astype(block.dtype)
                hits[zsi1 * hz:zsi1 * hz + dsblock.shape[0],
                ysi1 * hy:ysi1 * hy + dsblock.shape[1],
                xsi1 * hx:xsi1 * hx + dsblock.shape[2]] += 1
        block[hits > 0] = block[hits > 0] // hits[hits > 0]
        dest_directory.write_block(block, x0d[xidx], y0d[yidx], z0d[zidx])

    @staticmethod
    def write_stack_level_n(src_directory_id, dest_directory_id,
                          x0d, x0s, x1d, x1s, xidx, xsi_max,
                          y0d, y0s, y1d, y1s, yidx, ysi_max,
                          z0d, z0s, z1d, z1s, zidx, zsi_max):
        src_directory = directories[src_directory_id]
        dest_directory = directories[dest_directory_id]
        block = np.zeros((z1d[zidx] - z0d[zidx],
                          y1d[yidx] - y0d[yidx],
                          x1d[xidx] - x0d[xidx]), np.uint64)
        hits = np.zeros((z1d[zidx] - z0d[zidx],
                         y1d[yidx] - y0d[yidx],
                         x1d[xidx] - x0d[xidx]), np.uint64)
        hx = src_directory.x_block_size // 2
        hy = src_directory.y_block_size // 2
        hz = src_directory.z_block_size // 2
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
                block[zsi1 * hz:zsi1 * hz + dsblock.shape[0],
                      ysi1 * hy:ysi1 * hy + dsblock.shape[1],
                      xsi1 * hx:xsi1 * hx + dsblock.shape[2]] = np.maximum(
                    block[zsi1 * hz:zsi1 * hz + dsblock.shape[0],
                          ysi1 * hy:ysi1 * hy + dsblock.shape[1],
                          xsi1 * hx:xsi1 * hx + dsblock.shape[2]],
                    dsblock.astype(block.dtype))
        dest_directory.write_block(block, x0d[xidx], y0d[yidx], z0d[zidx])
