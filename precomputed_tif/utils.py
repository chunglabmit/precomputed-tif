import contextlib
import numpy as np
import os
import shutil
import tempfile
import tifffile
from .stack import Stack


@contextlib.contextmanager
def make_case(dtype, shape, return_path=False):
    """Make a test case

    :param dtype: the dtype of the volume, e.g. np.uint16
    :param shape: the shape of the volume
    :param return_path: if True, return the Glob expression, and destination,
           not the path.
    :return: the precomputed_tif.Stack and numpy 3-d volume that form the
             test case.
    """
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
        glob_expr = os.path.join(src, "img_*.tiff")
        stack = Stack(glob_expr, dest)
        if return_path:
            yield glob_expr, dest, npstack
        else:
            yield stack, npstack
    finally:
        shutil.rmtree(tempdir)
