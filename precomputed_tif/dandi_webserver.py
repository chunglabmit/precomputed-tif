"""Webserver module for mod_wsgi in Apache (e.g.) serving DANDI volumes

To use, create a virtualenv with precomputed-tif installed. Create a json
file containing a list of sources to serve. Each source should have a
"name" key, a "directory" key and a "format" key, e.g.:
[
   {
       "name": "expt1-dapi",
       "urls": [
          "file://path-to/foo_chunk-1_spim.ngff",
          "file://path-to/foo_chunk-2_spim.ngff"
            ]
   },
   {
       "name": "expt1-phalloidin",
       "urls": [
          "file://path-to/bar_chunk-1_spim.ngff",
          "file://path-to/bar_chunk-2_spim.ngff"
            ]
   }
]

Create a Python file for to serve via wsgi, e.g.

from precomputed_tif.dandi_webserver import serve_precomputed

CONFIG_FILE = "/etc/precomputed.config"

def application(environ, start_response):
    return serve_precomputed(environ, start_response, config_file)

"""

import json
import urllib

from .client import DANDIArrayReader


def file_not_found(dest, start_response):
    start_response("404 Not found",
                   [("Content-type", "text/html")])
    return [("<html><body>%s not found</body></html>" % dest).encode("utf-8")]


def serve_precomputed(environ, start_response, config_file):
    with open(config_file) as fd:
        config = json.load(fd)
    path_info = environ["PATH_INFO"]
    for source in config:
        if path_info[1:].startswith(source["name"]+"/"):
            filename = path_info[2+len(source["name"]):]
            urls = source["urls"]
            if filename == "info":
                return serve_info(environ, start_response, urls)
            else:
                level, x0, x1, y0, y1, z0, z1 = \
                    parse_filename(filename)
                ar = DANDIArrayReader(urls, level=level)
                img = ar[z0:z1, y0:y1, x0:x1]
                data = img.tostring("C")
                start_response(
                    "200 OK",
                    [("Content-type", "application/octet-stream"),
                     ("Content-Length", str(len(data))),
                     ('Access-Control-Allow-Origin', '*')])
                return [data]
    else:
        return file_not_found(path_info, start_response)


def serve_info(environ, start_response, urls):
    ar = DANDIArrayReader(urls, level=1)
    info_url = urls[0] + "/info"
    with urllib.request.urlopen(info_url) as fd:
        info = json.load(fd)
    for scale in info["scales"]:
        xs, ys, zs = [int(_) for _ in scale["key"].split("_")]
        scale["size"] = [int(ar.shape[2] // xs),
                         int(ar.shape[1] // ys),
                         int(ar.shape[0] // zs)]
    data = json.dumps(info, indent=2, ensure_ascii=True).encode("ascii")

    start_response(
        "200 OK",
        [("Content-type", "application/json"),
         ("Content-Length", str(len(data))),
         ('Access-Control-Allow-Origin', '*')])
    return [data]


def parse_filename(filename):
    level, path = filename.split("/")
    level = int(level.split("_")[0])
    xstr, ystr, zstr = path.split("_")
    x0, x1 = [int(x) for x in xstr.split('-')]
    y0, y1 = [int(y) for y in ystr.split('-')]
    z0, z1 = [int(z) for z in zstr.split('-')]
    return level, x0, x1, y0, y1, z0, z1


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