import sys
import argparse
import cv2 as cv
import numpy as np
from imblend import *

def read_image(filename, col=0):

    success = False
    img = cv.imread(filename, col).astype(np.float64)/255.

    msg = 'Image read failed'
    if not img is None:
        msg = "Success"
        success = True
    return success, msg, img


def write_image(filename, img):

    success = False
    msg = 'No Image Available'
    success = cv.imwrite(filename, np.uint8(img*255))
    if not success:
        msg = "Image write failed"

    return success, msg


def parse_arguments(argv, prog=''):
    # Initialize the command-line parser
    parser = argparse.ArgumentParser(prog,
                                     description='Script for imBlend.')

    #
    # Main input/output arguments
    #

    parser.add_argument('--source',
                        type=str,
                        help='Path to source image',
                        required=True)

    parser.add_argument('--target',
                        type=str,
                        help='Path to target image',
                        required=True)

    parser.add_argument('--mask',
                        type=str,
                        help='Path to mask image',
                        required=True)

    parser.add_argument('--output',
                        type=str,
                        help='Path to save the results',
                        default='output',
                        required=True)

    parser.add_argument('--levels',
                        type=int,
                        help='Number of convolutions used to process the image',
                        default=20,
                        required=False)

    parser.add_argument('--col',
                        action='store_true',
                        help='Set flag to true to produce color images',
                        required=False)


    # Run the python argument-parsing routine, leaving any
    #  unrecognized arguments intact
    args, unprocessed_argv = parser.parse_known_args(argv)

    # return any arguments that were not recognized by the parser
    return args, unprocessed_argv


def main(argv, prog=''):
    args, unprocessed_argv = parse_arguments(argv, prog)
    col = 0
    if(args.col):
        col = 1

    success, msg, src = read_image(args.source, col)
    if not success:
        print("Error: read_image: " + msg)
        exit(1)

    success, msg, mask = read_image(args.mask, col)
    if not success:
        print("Error: read_image: " + msg)
        exit(1)

    success, msg, tar = read_image(args.target, col)
    if not success:
        print("Error: read_image: " + msg)
        exit(1)

    out = args.output
    size = args.levels

    blended = None
    bad = None
    if(col):
        blendedR, blendedG, blendedB = imBlend(src[...,0], mask[...,0], tar[...,0], size), \
                                            imBlend(src[...,1], mask[...,1], tar[...,1], size), \
                                            imBlend(src[...,2], mask[...,2],tar[...,2], size)
        bad = np.stack([dummyPlace(src[...,0], mask[...,0], tar[...,0]), dummyPlace(src[...,1], \
                                            mask[...,1], tar[...,1]), dummyPlace(src[...,2], \
                                            mask[...,2], tar[...,2])], axis=2)
        blended = np.stack([blendedR, blendedG, blendedB], axis=2)
        write_image("./dummy.png", bad)
    else:
        blended = imBlend(src, mask, tar, size)
        write_image("./dummy.png", dummyPlace(src, mask, tar))

    success, msg = write_image(out, blended)
    if not success:
        print("Error: write_image: " + msg)

    print("Done.")


if __name__ == '__main__':
    main(sys.argv[1:], sys.argv[0])
