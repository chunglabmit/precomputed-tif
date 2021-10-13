# precomputed-tif
A simple utility to generate a precomputed data source as TIF files along with a simple HTTP server of those files.

[![Travis CI Status](https://travis-ci.com/chunglabmit/precomputed-tif.svg?branch=master)](https://travis-ci.com/chunglabmit/precomputed-tif)

## Installation

**precomputed-tif** has few dependencies. It can be installed using
the commands:

```commandline
pip install https://github.com/chunglabmit/blockfs/archive/master.zip
pip install https://github.com/chunglabmit/precomputed-tif/archive/master.zip
```

## Usage

**precomputed-tif** takes a z-stack of tif files as input. The stack
will be assembled in alphabetical order, so numerical filenames
should be used with zero-padding, e.g. "image_0000.tif". These
are assembled into a directory that can be referenced using
the *cors_webserver*.

The command-line is:

```commandline
precomputed-tif --source <source-glob> \
                --dest <destination-folder> \
                [--levels <levels>] \
                [--voxel-size <voxel-size>] \
                [--format <format>] \
                [--n-cores <n-cores>] \
                [--log <log-level>]
```

where
* **source-glob** is a "glob" expression that collects the files,
e.g. "/path/to/img_*.tiff"
* **destination-folder** is the name of the folder that will hold
the mipmap levels of the precomputed tifs.
* **levels** the number of mipmap levels. The first level has the
same resolution as the input. Each subsequent level has half the
resolution as the previous level.
* **voxel-size** is the voxel size in microns as three comma-separated values,
                 e.g. "1.8,1.8,2.0" in X, Y, Z order.  The default is
                 "1.8,1.8,2.0" which is the voxel size for 4x SPIM.
* **format** is "tiff", "zarr" or "blockfs" to store the image data blocks as
                 3D TIFF files, ZARR array blocks or in the blockfs format.
                 The default is "tiff",  but "blockfs" is the most scalable
                 for very large volumes.
* **n-cores** is the number of processes that will be used for parallel
                 processing.
* **log** is the logging level, one of "DEBUG", "INFO", "WARNING" or "ERROR".

### precomputed-write-points

**precomputed-write-points** writes a json-encoded set of points in x, y, z
format out to the precomputed neuroglancer format for points display. Note:
for Leviathan, if you use an output directory of 
"/mnt/beegfs/neuroglancer/points/*name*", Leviathan will serve it at the
url, "precomputed://https://leviathan-chunglab.mit.edu/points/*name*".

Usage:

precomputed-write-points \
  --input <points-file> \
  --output-directory <output-directory> \
  [--voxel-size <voxel-size>] \
  [--lower-bound <lower-bound>] \
  [--upper-bound <upper-bound>] \
  [--no-by-id]

where:
* **points-file** is the points .json file
* **output-directory** is the base directory for storing the neuroglancer
annotations
* **voxel-size** is the size of a voxel in microns, expressed as "x,y,z"
The default is 1.8,1.8.2.0
* **lower-bound** is the lower bound of the annotation volume in voxels,
expressed as "x,y,z". The default is 0,0,0
* **upper-bound** is the upper bound of the annotation volume in voxels,
expressed as "x,y,z". The default is the maximum value per coordinate in
the points file
* **--no-by-id** - providing this flag will prevent writing the by-id
annotations. These are one annotation per file and, if there are a large
number of points, there will be too many of them for a file-system.

### cors-webserver

The **precomputed-webserver** serves the precomputed TIF files as a
"precomputed:" data source for Neuroglancer. You can create a layer
within Neuroglancer with one of these precomputed URLs. The format is
"precomputed://http://<server-address>:<port>" where server-address
is typically "localhost", but can also be the IP address of your
machine. "port" is the port number of the precomputed-webserver.

The command-line for the cors-webserver is:

```commandline
precomputed-webserver \
  --port <port-number> \
  --bind <ip-address> \
  --directory <directory>
  --format tiff
```

where
* **port-number** is the port number that the webserver will use
to serve  the volume.
* **bind** is the bind IP address - typically "localhost", but
possibly the ip-address of one of the network cards if serving
outside of the machine
* **directory** is the directory created by **precomputed-webserver**

### Apache / wsgi webserver

The WSGI webserver serves precomputed datasources via the WSGI
interface. If you install mod_wsgi for Apache, then you can serve
a configurable set of precomputed datasources. To configure Apache,
install and enable mod_wsgi, e.g. for Ubuntu: 
```bash
$ sudo apt install libapache2-mod-wsgi-py3
```
Create a sources.json file. This is a list of dictionaries with each
dictionary representing a precomputed data source. There are 3 keys:

* **name**: The name to use in the precomputed URL, e.g. your experiment
and channel.
* **directory**: The root directory of the datasource
* **format**: either "tiff" or "zarr"

Here is a possible sources.json:
```json
[
   {
       "name": "expt1-dapi",
       "directory": "/data/expt1/dapi_precomputed",
       "format": "tiff"
   },
   {
       "name": "expt1-phalloidin",
       "directory": "/data/expt1/phalloidin_precomputed",
       "format": "zarr"
   }
]
```

Next, create a WSGI script that points at both the wsgi_webserver and
the sources.json file. A WSGI script is a Python file with a single
function or class (see https://www.python.org/dev/peps/pep-0333/).
This file should be created in an empty directory which will be served
by mod_wsgi.

For instance:
```python
from precomputed_tif.wsgi_webserver import serve_precomputed

CONFIG_FILE = "/etc/sources.json"

def application(environ, start_response):
    return serve_precomputed(environ, start_response, CONFIG_FILE)
```

Finally, configure the webserver to serve the file. See
https://modwsgi.readthedocs.io/en/develop/index.html for directions
on how to do this. Here's a quick config file example that works and
may provide a rough template:
```text
WSGIDaemonProcess precomputed python-home=<path-to>/anaconda3/envs/precomputed-tif
WSGIProcessGroup precomputed
WSGIApplicationGroup %{GLOBAL}
<Directory /usr/local/apache-scripts>
<IfVersion < 2.4>
    Order allow,deny
    Allow from all
</IfVersion>
<IfVersion >= 2.4>
    Require all granted
</IfVersion>
</Directory>
WSGIScriptAlias /precomputed /usr/local/apache-scripts/precomputed.wsgi
```
With the above, you should be able to serve precomputed datasources
at precomputed://http://localhost/precomputed/expt1-dapi, etc.
