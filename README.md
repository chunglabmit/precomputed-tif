# precomputed-tif
A simple utility to generate a precomputed data source as TIF files along with a simple HTTP server of those files.

## Installation

**precomputed-tif** has few dependencies. It can be installed using
the command:

```commandline
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
                --levels <levels>

```

where
* **source-glob** is a "glob" expression that collects the files,
e.g. "/path/to/img_*.tiff"
* **destination-folder** is the name of the folder that will hold
the mipmap levels of the precomputed tifs.
* **levels** the number of mipmap levels. The first level has the
same resolution as the input. Each subsequent level has half the
resolution as the previous level.

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