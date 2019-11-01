#!/usr/bin/env python
# @license
# Copyright 2017 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple web server serving local files that permits cross-origin requests.

This can be used to view local data with Neuroglancer.

WARNING: Because this web server permits cross-origin requests, it exposes any
data in the directory that is served to any web page running on a machine that
can connect to the web server.
"""

from __future__ import print_function, absolute_import

import argparse
from blockfs.directory import Directory
import os
import sys
import io
import zarr
import numpy as np
import pathlib

FORMAT_RAW = "raw"
FORMAT_TIFF = "tiff"
FORMAT_ZARR = "zarr"
FORMAT_BLOCKFS = "blockfs"

args = None

try:
    # Python3 and Python2 with future package.
    from http.server import SimpleHTTPRequestHandler, HTTPServer, HTTPStatus
except ImportError:
    from BaseHTTPServer import HTTPServer, HTTPStatus
    from SimpleHTTPServer import SimpleHTTPRequestHandler


class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if args.format == FORMAT_RAW or self.path.startswith("/socket"):
            print("Handing request to simple http request handler")
            super(RequestHandler, self).do_GET()
        elif self.path.find("/info") >= 0:
            size = pathlib.Path("info").stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", 'application/octet-stream')
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open("info", "rb") as fd:
                self.copyfile(fd, self.wfile)

        elif args.format == FORMAT_TIFF:
            import tifffile
            path = self.path[1:] + ".tiff"
            print(path)
            if not os.path.exists(path):
                super(RequestHandler, self).do_GET()
                return
            chunk = tifffile.imread(path)
            byteorder = "C"
            data = chunk.tostring(byteorder)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", 'application/octet-stream')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.copyfile(io.BytesIO(data), self.wfile)
        elif args.format == FORMAT_ZARR:
            import zarr
            level, path = self.path[1:].split('/')
            if not os.path.exists(level):
                super(RequestHandler, self).do_GET()
                return
            xstr, ystr, zstr = path.split('_')
            x0, x1 = [int(x) for x in xstr.split('-')]
            y0, y1 = [int(y) for y in ystr.split('-')]
            z0, z1 = [int(z) for z in zstr.split('-')]
            store = zarr.NestedDirectoryStore(level)
            z_arr = zarr.open(store, mode='r')
            chunk = z_arr[z0:z1, y0:y1, x0:x1]
            byteorder = "C"
            data = chunk.tostring(byteorder)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", 'application/octet-stream')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.copyfile(io.BytesIO(data), self.wfile)
        elif args.format == FORMAT_BLOCKFS:
            level, path = self.path[1:].split('/')
            if not os.path.exists(level):
                super(RequestHandler, self).do_GET()
                return
            xstr, ystr, zstr = path.split('_')
            x0, x1 = [int(x) for x in xstr.split('-')]
            y0, y1 = [int(y) for y in ystr.split('-')]
            z0, z1 = [int(z) for z in zstr.split('-')]
            directory = Directory.open(
                os.path.join(level, "precomputed.blockfs"))
            chunk = directory.read_block(x0, y0, z0)
            byteorder = "C"
            data = chunk.tostring(byteorder)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", 'application/octet-stream')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.copyfile(io.BytesIO(data), self.wfile)
        else:
            raise ValueError('Invalid format specified')

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)


class Server(HTTPServer):
    protocol_version = 'HTTP/1.1'

    def __init__(self, server_address):
        HTTPServer.__init__(self, server_address, RequestHandler)


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--port', type=int, default=9000, help='TCP port to listen on')
    ap.add_argument('-a', '--bind', default='127.0.0.1', help='Bind address')
    ap.add_argument('-d', '--directory', default='.', help='Directory to serve')
    ap.add_argument('-f', '--format', default=FORMAT_RAW,
                    help="Format of the backend volumes: raw, tiff or zarr")

    args = ap.parse_args()
    os.chdir(args.directory)
    server = Server((args.bind, args.port))
    sa = server.socket.getsockname()
    print("Serving directory %s at http://%s:%d" % (os.getcwd(), sa[0], sa[1]))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        sys.exit(0)


if __name__ == "__main__":
    main()
