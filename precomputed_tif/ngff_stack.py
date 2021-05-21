import itertools
import multiprocessing

import numcodecs
import numpy as np
import tifffile
import tqdm
import typing

from scipy import ndimage

from .stack import StackBase, PType
import pathlib
import scipy
import uuid
import zarr

DATASETS:typing.Mapping[uuid.UUID, zarr.Array] = {}


class NGFFStack(StackBase):
    """
    Implements the NGFF specification:
    https://ngff.openmicroscopy.org/0.2/

    """

    VERSION=0.2
    TYPE="reduce"

    def __init__(self, *args, **kwargs):
        super(NGFFStack, self).__init__(*args, **kwargs)
        path = pathlib.Path(self.dest)
        self.name = path.stem
        self.chunksize = (64, 64, 64)

    def create(self, mode="w", compressor=numcodecs.Blosc("zstd", 5)):
        """
        Create or open for append a dataset

        :param n_channels: # of channels in the zarr
        :param current_channel: the channel to be written
        """
        store = zarr.NestedDirectoryStore(
            self.dest)
        self.zgroup = zarr.group(store,
                                 overwrite=(mode == "w"))
        self.compressor = compressor

    def write_info_file(self,
                        n_levels,
                        voxel_size=(1800, 1800, 2000)):
        """
        Write the level metadata for the spec as well as
        the info file
        """
        super(NGFFStack, self).write_info_file(n_levels, voxel_size)

        self.zgroup.attrs["multiscales"] = [dict(
                version=self.VERSION,
                name=self.name,
                type=self.TYPE,
                metadata=dict(
                    method="scipy.ndimage.map_coordinates",
                    version=scipy.__version__
                ),
                datasets=[dict(path=str(i)) for i in range(n_levels)]

            )]
        self.zgroup.attrs["omero"]=dict(
                name=self.name,
                version=self.VERSION,
                channels=[
                    dict(
                        active=True,
                        coefficient=1,
                        color="FFFFFF",
                        family="linear",
                        inverted=False,
                        label=self.name,
                        window=dict(
                            min=np.iinfo(self.dtype).min,
                            max=np.iinfo(self.dtype).max
                        )
                    )
                ],
                rdefs=dict(
                    defaultT=0,
                    defaultZ=self.z_extent // 2,
                    model="greyscale"
                )
            )

    def write_level_1(self, silent=False,
                      n_cores=min(multiprocessing.cpu_count(), 4)):
        z0 = self.z0(1)
        z1 = self.z1(1)
        y0 = self.y0(1)
        y1 = self.y1(1)
        x0 = self.x0(1)
        x1 = self.x1(1)
        x_extent = self.x_extent
        y_extent = self.y_extent
        dataset = self.zgroup.create_dataset(
            "0",
            shape=(1, 1, self.z_extent, y_extent, x_extent),
            chunks=(1, 1, self.cz(), self.cy(), self.cx()),
            dtype=self.dtype,
            compressor=self.compressor,
            fill_value=0,
            overwrite=True
        )
        my_id = uuid.uuid4()
        try:
            DATASETS[my_id] = dataset
            with multiprocessing.Pool(n_cores) as pool:
                futures = []
                for z0a, z1a in zip(z0, z1):
                    files = self.files[z0a:z1a]
                    futures.append(pool.apply_async(
                        self.write_one_level_1,
                        (my_id, files, x0, x1, y0, y1, z0a, z1a)))
                for future in tqdm.tqdm(futures, disable=silent):
                    future.get()
        finally:
            del DATASETS[my_id]

    @staticmethod
    def write_one_level_1(dataset_id,
            files, x0, x1, y0, y1, z0, z1):
        dataset = DATASETS[dataset_id]
        y_extent, x_extent = dataset.shape[-2:]
        img = np.zeros((1, 1, z1-z0, y_extent, x_extent), dataset.dtype)
        for z, file in zip(range(z0, z1), files):
            img[0, 0, z - z0] = \
                tifffile.imread(file)
        for (x0a, x1a), (y0a, y1a) in itertools.product(
                zip(x0, x1), zip(y0, y1)):
            dataset[:, :, z0:z1, y0a:y1a, x0a:x1a] = \
                img[:, :, :, y0a:y1a, x0a:x1a]

    def write_level_n(self, level,
                      n_cores=min(multiprocessing.cpu_count(), 13)):
        src_dataset = self.zgroup[str(level - 2)]
        x_extent, y_extent, z_extent = [
            (extent + self.resolution(level) - 1) //  self.resolution(level)
            for extent in (self.x_extent, self.y_extent, self.z_extent)]
        dest_dataset = self.zgroup.create_dataset(
            str(level-1),
            shape=(1, 1, z_extent, y_extent, x_extent),
            chunks=(1, 1, self.cz(), self.cy(), self.cx()),
            dtype=self.dtype,
            compressor=self.compressor,
            fill_value=0,
            overwrite=True
        )
        src_id = uuid.uuid4()
        dest_id = uuid.uuid4()
        DATASETS[src_id] = src_dataset
        DATASETS[dest_id] = dest_dataset
        try:
            with multiprocessing.Pool(n_cores) as pool:
                futures = []
                for (x0, x1), (y0, y1), (z0, z1) in \
                    itertools.product(zip(self.x0(level), self.x1(level)),
                                      zip(self.y0(level), self.y1(level)),
                                      zip(self.z0(level), self.z1(level))):
                    futures.append(pool.apply_async(
                        NGFFStack.write_one_level_n,
                        (src_id, dest_id, x0, x1, y0, y1, z0, z1, self.ptype)
                    ))
                for future in tqdm.tqdm(futures,
                                        desc="Writing level %d" % level):
                    future.get()
        finally:
            del DATASETS[src_id]
            del DATASETS[dest_id]

    @staticmethod
    def write_one_level_n(src_id, dest_id, x0, x1, y0, y1, z0, z1, ptype):
        src_dataset = DATASETS[src_id]
        dest_dataset = DATASETS[dest_id]
        z1s, y1s, x1s = [min(a * 2, b) for a, b in zip((z1, y1, x1),
                                                        src_dataset.shape[2:])]
        src_block = src_dataset[0, 0, z0 * 2:z1s, y0 * 2: y1s, x0 * 2: x1s]
        if np.all(src_block == 0):
            # Don't bother writing out an empty slot
            return
        zs,ys,xs = np.mgrid[0:z1-z0, 0:y1-y0, 0:x1-x0] * 2 + .5
        order = 0 if ptype == PType.SEGMENTATION else 1
        dest_block = ndimage.map_coordinates(src_block, (zs, ys, xs),
                                             order=order,
                                             mode="nearest")
        dest_dataset[0, 0, z0:z1, y0:y1, x0:x1] = dest_block
