from setuptools import setup

version = "0.1.0"

with open("./README.md") as fd:
    long_description = fd.read()

setup(
    name="precomputed-tif",
    version=version,
    description=
    "Make and serve tiffs as neuroglancer precomputed data source",
    long_description=long_description,
    install_requires=[
        "blockfs",
        "numpy",
        "numcodecs",
        "requests",
        "tifffile",
        "tqdm",
        "zarr"
    ],
    author="Kwanghun Chung Lab",
    packages=["precomputed_tif"],
    entry_points={ 'console_scripts': [
        'precomputed-tif=precomputed_tif.main:main',
        'precomputed-webserver=precomputed_tif.cors_webserver:main'
    ]},
    url="https://github.com/chunglabmit/precomputed-tif",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.5',
    ],
)