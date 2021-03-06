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


import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import os
import numpy as np
import cv2
import pandas as pd

from Logging import log
import utils


class DatasetReader(Dataset):
    """
    Dataset reader
    """

    def __init__(self, input_list, target_list, transformer=None):

        self.transformer = transformer

        # with open(datasetInputFile, 'r') as fList:
        #     lines = [line.strip() for line in fList.readlines()]
        #     input_len = len(lines)
        #     self.input = lines

        # with open(datasetTargetFile, 'r') as fList:
        #     lines = [line.strip() for line in fList.readlines()]
        #     target_len = len(lines)
        #     self.target = lines

        self.input = input_list
        self.target = target_list

        if len(self.input) != len(target_list):
            raise Exception(
                "The number of input images mismatch the number of target images")
            self.dataset_len = -1
        else:
            self.dataset_len = len(self.input)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):

        # Read images
        X_np = cv2.imread(self.input[index])
        Y_np = cv2.imread(self.target[index])

        # Transform image from [0-255] to [0,1]
        X_np = X_np.astype(np.float)/255.0
        Y_np = Y_np.astype(np.float)/255.0

        # Swap channels
        X_np = np.rollaxis(X_np, 2, 0)
        Y_np = np.rollaxis(Y_np, 2, 0)

        # Transform images
        if self.transformer is not None:
            X_np = self.transformer(X_np)
            Y_np = self.transformer(Y_np)

        # Create tensors
        X_tensor = torch.from_numpy(X_np)
        Y_tensor = torch.from_numpy(Y_np)
        # print(X_tensor.type())

        return X_tensor, Y_tensor

    def get_image_shape(self):
        X, Y = self[0]
        return X.size(), Y.size()


def main():

    input_list = utils.get_image_list("/media/rodrigo/c6e79420-ed98-4730-96f6-ff18c884341d/Rodrigo/Unicamp/IA368Z/trab-final/dataset/300x300_cr90")
    target_list = utils.get_image_list("/media/rodrigo/c6e79420-ed98-4730-96f6-ff18c884341d/Rodrigo/Unicamp/IA368Z/trab-final/dataset/300x300_cr0")

    dataset = DatasetReader(input_list,target_list)

    log.info("Dataset size: {}".format(len(dataset)))

    log.info("Accessing single sample")
    log.info(dataset[0][0].shape)
    log.info(dataset[0][1].shape)
    log.info(dataset[0][0][0, 0, 0])

    log.info("Accessing via Dataloader")

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True)

    log.info(dataset.get_image_shape())

    # for batch in train_loader:
    #     x_train, y_train = batch
    #     log.info(x_train.shape)
    #     log.info(y_train.shape)


if __name__ == "__main__":
    main()
