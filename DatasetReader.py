#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
import pandas as pd

from Logging import log


image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'gif')


class DatasetReader(Dataset):
    """
    Dataset reader
    """

    def __init__(self, datasetInputFile, datasetTargetFile, transformer=None):

        self.transformer = transformer

        with open(datasetInputFile, 'r') as fList:
            lines = [line.strip() for line in fList.readlines()]
            input_len = len(lines)
            self.input = lines

        with open(datasetTargetFile, 'r') as fList:
            lines = [line.strip() for line in fList.readlines()]
            target_len = len(lines)
            self.target = lines

        if input_len != target_len:
            raise Exception(
                "The number of input images mismatch the number of target images")
            self.dataset_len = -1
        else:
            self.dataset_len = input_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        X_np = cv2.imread(self.input[index])
        Y_np = cv2.imread(self.target[index])

        if self.transformer is not None:
            X_np = self.transformer(X_np)
            Y_np = self.transformer(Y_np)

        X_tensor = torch.from_numpy(X_np)
        Y_tensor = torch.from_numpy(Y_np)

        return X_tensor, Y_tensor


def main():
    dataset = DatasetReader(
        "/home/rodrigo/unicamp/IA368Z/trab-final/dataset/input_dataset.txt",
        "/home/rodrigo/unicamp/IA368Z/trab-final/dataset/target_dataset_2x.txt")

    log.info("Dataset size: {}".format(len(dataset)))

    log.info("Accessing single sample")
    log.info(dataset[0][0].shape)
    log.info(dataset[0][1].shape)

    log.info("Accessing via Dataloader")

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True)

    for batch in train_loader:
        x_train, y_train = batch
        print(x_train.shape)
        print(y_train.shape)

if __name__ == "__main__":
    main()
