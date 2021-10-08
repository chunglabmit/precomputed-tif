"""Webserver module for mod_wsgi in Apache (e.g.)

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

"""

import json
import os
import pathlib

import numpy as np


def file_not_found(dest, start_response):
    start_response("404 Not found",
                   [("Content-type", "text/html")])
    return [("<html><body>%s not found</body></html>" % dest).encode("utf-8")]


class ParseFilenameError(BaseException):
    pass


def serve_precomputed(environ, start_response, config_file):
    with open(config_file) as fd:
        config = json.load(fd)
    path_info = environ["PATH_INFO"]
    for source in config:
        if path_info[1:].startswith(source["name"]+"/"):
            try:
                filename = path_info[2+len(source["name"]):]
                dest = os.path.join(source["directory"])
                if filename == "mesh/info":
                    return file_not_found(filename, start_response)
                elif filename == "info":
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
                    filename, x0, x1, y0, y1, z0, z1 = \
                        parse_filename(dest, filename)
                    if not os.path.exists(filename):
                        return file_not_found(filename, start_response)
                    store = zarr.NestedDirectoryStore(filename)
                    z_arr = zarr.open(store, mode='r')
                    chunk = z_arr[z0:z1, y0:y1, x0:x1]
                    data = chunk.tostring("C")
                    start_response(
                        "200 OK",
                        [("Content-type", "application/octet-stream"),
                         ("Content-Length", str(len(data))),
                         ('Access-Control-Allow-Origin', '*')])
                    return [data]
                elif source["format"] == "blockfs":
                    from blockfs import Directory
                    filename, x0, x1, y0, y1, z0, z1 = \
                        parse_filename(dest, filename)
                    filename = os.path.join(filename, "precomputed.blockfs")
                    directory = Directory.open(filename)
                    chunk = directory.read_block(x0, y0, z0)
                    data = chunk.tostring("C")
                    start_response(
                        "200 OK",
                        [("Content-type", "application/octet-stream"),
                         ("Content-Length", str(len(data))),
                         ('Access-Control-Allow-Origin', '*')])
                    return [data]
                elif source["format"] == "ngff":
                    import zarr
                    filename, x0, x1, y0, y1, z0, z1 = \
                        parse_filename(dest, filename)
                    root, level = os.path.split(filename)
                    lx, ly, lz = [int(_) for _ in level.split("_")]
                    llevel = int(np.round(np.log2(lx), 0))
                    store = zarr.NestedDirectoryStore(root)
                    group = zarr.group(store)
                    a = group[llevel]
                    _, _, zs, ys, xs = a.chunks
                    z1 = min(a.shape[2], z0 + zs)
                    y1 = min(a.shape[3], y0 + ys)
                    x1 = min(a.shape[4], x0 + xs)
                    chunk = a[0, 0, z0:z1, y0:y1, x0:x1]
                    data = chunk.tostring("C")
                    start_response(
                        "200 OK",
                        [("Content-type", "application/octet-stream"),
                         ("Content-Length", str(len(data))),
                         ('Access-Control-Allow-Origin', '*')])
                    return [data]
            except ParseFilenameError:
                return file_not_found(path_info, start_response)
    else:
        return file_not_found(path_info, start_response)


def parse_filename(dest, filename):
    try:
        level, path = filename.split("/")
        filename = os.path.join(dest, level)
        xstr, ystr, zstr = path.split("_")
        x0, x1 = [int(x) for x in xstr.split('-')]
        y0, y1 = [int(y) for y in ystr.split('-')]
        z0, z1 = [int(z) for z in zstr.split('-')]
    except ValueError:
        raise ParseFilenameError()
    return filename, x0, x1, y0, y1, z0, z1

if __name__ == "__main__":
    from wsgiref.simple_server import make_server
    import sys

    config_filename = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    else:
        port = 8080


    def application(environ, start_response):
        return serve_precomputed(environ, start_response, config_filename)


    httpd = make_server("127.0.0.1", port, application)
    httpd.serve_forever()