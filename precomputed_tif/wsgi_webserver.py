'''Webserver module for mod_wsgi in Apache (e.g.)

To use, create a virtualenv with precomputed-tif installed. Create a json
file containing a list of sources to serve. Each source should have a
"name" key, a "directory" key and a "format" key, e.g.:
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

Create a Python file for to serve via wsgi, e.g.

from precomputed_tif.wsgi_webserver import serve_precomputed

CONFIG_FILE = "/etc/precomputed.config"

def application(environ, start_response):
    return serve_precomputed(environ, start_response, config_file)

'''

import json
import os
import pathlib

def file_not_found(dest, start_response):
    start_response("404 Not found",
                   [("Content-type", "text/html")])
    return [("<html><body>%s not found</body></html>" % dest).encode("utf-8")]

def serve_precomputed(environ, start_response, config_file):
    with open(config_file) as fd:
        config = json.load(fd)
    path_info = environ["PATH_INFO"]
    for source in config:
        if path_info[1:].startswith(source["name"]):
            filename = path_info[2+len(source["name"]):]
            dest = os.path.join(source["directory"])
            if filename == "info":
                destpath = os.path.join(dest, "info")
                if not os.path.exists(destpath):
                    return file_not_found(destpath, start_response)
                with open(destpath, "rb") as fd:
                    data = fd.read()
                start_response(
                    "200 OK",
                    [("Content-type", "application/json"),
                     ("Content-Length", str(len(data))),
                     ('Access-Control-Allow-Origin', '*')])
                return [data]
            elif source["format"] == "tiff":
                import tifffile
                path = os.path.join(dest, filename+".tiff")
                if not os.path.exists(path):
                    return file_not_found(path, start_response)
                img = tifffile.imread(path)
                data = img.tostring("C")
                start_response(
                    "200 OK",
                    [("Content-type", "application/octet-stream"),
                     ("Content-Length", str(len(data))),
                     ('Access-Control-Allow-Origin', '*')])
                return [data]
            elif source["format"] == "zarr":
                import zarr
                level, path = filename.split("/")
                filename = os.path.join(dest, level)
                if not os.path.exists(filename):
                    return file_not_found(filename, start_response)
                xstr, ystr, zstr = filename.split("_")
                x0, x1 = [int(x) for x in xstr.split('-')]
                y0, y1 = [int(y) for y in ystr.split('-')]
                z0, z1 = [int(z) for z in zstr.split('-')]
                store = zarr.NestedDirectoryStore(level)
                z_arr = zarr.open(store, mode='r')
                chunk = z_arr[z0:z1, y0:y1, x0:x1]
                data = chunk.tostring("C")
                start_response(
                    "200 OK",
                    [("Content-type", "application/octet-stream"),
                     ("Content-Length", str(len(data))),
                     ('Access-Control-Allow-Origin', '*')])
                return [data]
    else:
        return file_not_found(path_info, start_response)
