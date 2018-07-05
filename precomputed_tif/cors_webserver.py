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
import os
import sys
import io

FORMAT_RAW = "raw"
FORMAT_TIFF = "tiff"
FORMAT_ZARR = "zarr"

try:
    # Python3 and Python2 with future package.
    from http.server import SimpleHTTPRequestHandler, HTTPServer, HTTPStatus
except ImportError:
    from BaseHTTPServer import HTTPServer, HTTPStatus
    from SimpleHTTPServer import SimpleHTTPRequestHandler


class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if args.format == FORMAT_RAW or self.path == "/info":
            super(RequestHandler, self).do_GET()
        elif args.format == FORMAT_TIFF:
            import tifffile
            path = self.path[1:] + ".tiff"
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
        else:
            import zarr
            pass

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)


class Server(HTTPServer):
    protocol_version = 'HTTP/1.1'

    def __init__(self, server_address):
        HTTPServer.__init__(self, server_address, RequestHandler)


def main():
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
