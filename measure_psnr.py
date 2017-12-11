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

from utils import get_image_list
from SuperResNet import SuperResNet, SuperResNetVGG16
from skimage.measure import compare_psnr
import pytorch_trainer as pytt
from Logging import log
import numpy as np
import argparse
from DatasetReader import DatasetReader
from torch.utils.data import DataLoader



def measure_psnr(batch_size,input_dir,target_dir,model_path):
    
    # Create model
    #srn = SuperResNetVGG16()
    #srn = SuperResNet().double()
    srn = torch.nn.DataParallel(SuperResNetVGG16(_pretrained=False)).cuda()

    # Read input and target data into a DataLoader
    input_list = get_image_list(input_dir)
    target_list = get_image_list(target_dir)
    dataset = DatasetReader(input_list, target_list, None)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # Load network
    metrics = dict(train=dict(losses=[]), valid=dict(losses=[]))
    pytt.load_trainer_state(args.model_path, srn, metrics)
    srn.eval()

    psnr_array_X = []
    psnr_array_Y_ = []

    for curr_batch, (X, Y) in enumerate(train_loader):

        predict = pytt.predict(srn, X).numpy()
        psnr_array_Y_.append(compare_psnr(Y.numpy(),predict))
        psnr_array_X.append(compare_psnr(Y.numpy(),X.numpy()))
        log.info("Batch {} Ok".format(curr_batch+1))

    psnr_array_Y_ = np.array(psnr_array_Y_)
    psnr_array_X = np.array(psnr_array_X)
    log.info("PSNR Prediction -> Mean: {} Std: {}".format(np.mean(psnr_array_Y_),np.std(psnr_array_Y_)))
    log.info("PSNR Input -> Mean: {} Std: {}".format(np.mean(psnr_array_X),np.std(psnr_array_X)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="SuperResNet", description="Measure Peak Signal to Noise Ratio")
    parser.add_argument('--batch_size', help='Batch size',
                        type=int, required=True)
    parser.add_argument('--input_dir', help='Input list',
                        type=str, required=True)
    parser.add_argument('--target_dir', help='Target list',
                        type=str, required=True)
    parser.add_argument('--model_path', help='Path to save the model',
                        type=str, required=True)

    args = parser.parse_args()

    measure_psnr(**args.__dict__)