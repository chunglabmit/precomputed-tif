# imports
from precomputed_tif.client import ArrayReader
import pathlib
import tifffile
import math
from functools import partial
import argparse
import multiprocessing
parser = argparse.ArgumentParser()
parser.add_argument('--input-file-path', help='path of input file', required= True)
parser.add_argument('--output-file-path', help='path of output file', required= True)
parser.add_argument('--chunk', help='The size of the chunk in the z axis that you want to be processed each round. ', type=int, required= True)
parser.add_argument('--y0', help='y0 value', type=int, required= True)
parser.add_argument('--y1', help='y1 value', type=int, required= True)
parser.add_argument('--x0', help='x0 value', type=int, required= True)
parser.add_argument('--x1', help='x1 value', type=int,required= True)
args = parser.parse_args()

def subset_image_mult(input_file_path, output_file_path, chunk, y0, y1 , x0,x1, iteration):  # inputs: input file path, output file path, How much to process in z per call, x and y values, number of calls
    path = pathlib.Path("{file_path}".format(file_path=input_file_path))
    # ArrayReader takes the URL form of the path and it points to a blockfs volume
    array_reader = ArrayReader(path.as_uri(), format="blockfs")
    z_y_x_dim = array_reader.shape  # get the shape of the array : [640, 2542, 3000]
    z_dim= z_y_x_dim[0] # get the z dimension -> 640
    max_num_iterations= math.ceil(z_dim/chunk)# calculate the number of calls needed to go through all of the values in the z dimension
    if iteration <= max_num_iterations: # Only if the number of calls is reasonable proceed to the next step
        chunk_size=chunk
        start= z_dim - (z_dim- (chunk_size*(iteration-1)) ) # calculate what index to start on given the number of calls
        end = start+ chunk_size  #calculate what index to end on given the number of calls
        block= array_reader[start:end, y0:y1, x0:x1]  # create the block given the call size and chunk size for example chunk 64 and call 1 would be 0 to 63 and call 2 would start at z=64 and end at z=127
        ind=list(range(start,end)) # create a list from which to grab the indicies of the file names from
        for i,j in enumerate(block):
            tifffile.imsave("{output_path}/image_{index_number_z:04d}.tiff".format(output_path=output_file_path,index_number_z=ind[i]), j)
    else:
        print(f" The max number of iterations is {max_num_iterations}")

def main():
    # apply multiprocessing
    p = pathlib.Path("{file_path}".format(file_path=args.input_file_path))
    # ArrayReader takes the URL form of the path and it points to a blockfs volume
    ar = ArrayReader(p.as_uri(), format="blockfs")
    dim = ar.shape
    z_dimension = dim[0]
    max_iterations = math.ceil(z_dimension / 64)
    with multiprocessing.Pool() as pool:
        futures = []
        for z in range(1, max_iterations + 1):
            futures.append(pool.apply_async(
                partial(subset_image_mult, args.input_file_path, args.output_file_path, args.chunk, args.y0, args.y1,
                        args.x0, args.x1), (z,)))
        for future in futures:
            future.get()
if __name__ == "__main__":
    main()
