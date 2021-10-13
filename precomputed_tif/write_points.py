import argparse
import numpy as np
import json
import pathlib
import struct
import sys

# Some ideas here borrowed from
# https://githubmemory.com/repo/google/neuroglancer/issues/280

DESCRIPTION="write-points writes a Neuroglancer data source for" \
            " a set of points."

def parse_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--input",
        help="The JSON file containing a list of points in x,y,z format",
        required=True
    )
    parser.add_argument(
        "--output-directory",
        help="The directory in which the data source is stored",
        required=True
    )
    parser.add_argument(
        "--voxel-size",
        help="The size of a voxel in microns, in format \"x,y,z\"",
        default="1.8,1.8,2.0"
    )
    parser.add_argument(
        "--lower-bound",
        help="The lower bound of the points volume, in format \"x,y,z\".",
        default="0,0,0"
    )
    parser.add_argument(
        "--upper-bound",
        help="The upper bound of the points volume, in format \"x,y,z\"."
    )
    parser.add_argument(
        "--no-by-id",
        action="store_true",
        help="Add this switch to prevent writing the individual point files. "
        "This may be necessary if you have a large number of points"
    )
    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    opts = parse_arguments(args)
    with open(opts.input) as fd:
        points = np.array(json.load(fd))
    output_directory = pathlib.Path(opts.output_directory)
    if not output_directory.exists():
        output_directory.mkdir()
    lower_bound = [int(_) for _ in opts.lower_bound.split(",")]
    if opts.upper_bound:
        upper_bound = [int(_) for _ in opts.upper_bound.split(",")]
    else:
        upper_bound = [int(_) for _ in points.max(0)]
    voxel_size = [float(_) for _ in opts.voxel_size.split(",")]
    #
    # The info file
    #
    info = {
        "@type": "neuroglancer_annotations_v1",
        "annotation_type": "POINT",
        "by_id": { "key": "by_id" },
        "dimensions": {
            "x": [voxel_size[0], "um"],
            "y": [voxel_size[1], "um"],
            "z": [voxel_size[2], "um"]
        },
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "properties": [],
        "relationships": [],
        "spatial": [
            {
                "chunk_size": (points.max(0) + 1).astype(int).tolist(),
                "grid_shape": [1, 1, 1],
                "key": "spatial0",
                "limit": len(points)+1
            }
        ]
    }
    info_path = output_directory / "info"
    with info_path.open("w") as fd:
        json.dump(info, fd, indent=2)
    #
    # The single spatial file
    # TODO: add options for multiple spatial blocks and different
    #       spatial scales
    #
    # The format is
    # <# of points>
    # <point-block>
    # <# of ids>
    # <id-block>
    # where point-block is x, y, z packed as floats
    #       id-block is int64 ids for each point
    #
    spatial_path = output_directory / "spatial0" / "0_0_0"
    if not spatial_path.parent.exists():
        spatial_path.parent.mkdir()
    with spatial_path.open('wb') as fd:
        total_points = len(points)
        buffer = struct.pack('<Q', total_points)
        for (x, y, z) in points:
            annotpoint = struct.pack('<3f', x, y, z)
            buffer += annotpoint
        pointid_buffer = struct.pack('<%sQ' % len(points), *range(len(points)))
        buffer += pointid_buffer
        fd.write(buffer)
    #
    # The individual points, each gets a file (yuk)
    #
    if not opts.no_by_id:
        by_id_path = output_directory / "by_id"
        if not by_id_path.exists():
            by_id_path.mkdir()
        for idfileidx in range(len(points)):
            filename = by_id_path / str(idfileidx)
            with filename.open('wb') as fd:
                x = points[idfileidx, 0]
                y = points[idfileidx, 1]
                z = points[idfileidx, 2]
                annotpoint = struct.pack('<3f', x, y, z)
                fd.write(annotpoint)


if __name__=="__main__":
    main()
