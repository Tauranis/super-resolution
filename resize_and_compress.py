#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MIT License

# Copyright (c) 2017 Tauranis

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import os

import argparse
from Logging import log


def resize_and_compress(input_path, output_path, width, height, compression_rate):
    """
    Resize and compress image

    Params
    --------

    input_path : str
    output_path : str
    width : int
    height : int
    compression_rate : int

    """
    img = cv2.imread(input_path)
    img_res = cv2.resize(img, (width, height))
    cv2.imwrite(output_path, img_res, params=[
                cv2.IMWRITE_JPEG_QUALITY, 100 - compression_rate])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Resize and compress image')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--cr', type=int, required=True,
                        help="Compression rate [0-100]. The higher the more compressed.")

    args = parser.parse_args()

    # log.info("Input path: {}".format(args.input_path))
    # log.info("Output path: {}".format(args.output_path))
    # log.info("(width,height): ({},{})".format(args.width, args.height))
    # log.info("Compression rate: {}%".format(args.cr))

    if args.input_path.lower().endswith("jpg"):
        resize_and_compress(args.input_path, args.output_path, args.width, args.height, args.cr)
