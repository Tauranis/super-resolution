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

from DatasetReader import DatasetReader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.optim as optim
from SuperResNet import SuperResNet
import argparse
from Logging import log
from utils import EXTENSIONS, get_image_list

import pytorch_trainer as pytt
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="SuperResNet", description="Test Super Resolution CNN")
    parser.add_argument('--batch_size', help='Batch size',
                        type=int, required=True)
    parser.add_argument('--input_dir', help='Input list',
                        type=str, required=True)
    parser.add_argument('--target_dir', help='Target list',
                        type=str, required=True)
    parser.add_argument('--model_path', help='Path to save the model',
                        type=str, required=True)

    args = parser.parse_args()

    log.info("Configs")
    log.info("Batch Size: {}".format(args.batch_size))
    log.info("Input List: {}".format(args.input_dir))
    log.info("Target List: {}".format(args.target_dir))
    log.info("Model Path: {}".format(args.model_path))

    input_list = get_image_list(args.input_dir)
    target_list = get_image_list(args.target_dir)

    srn = SuperResNet(train=False)
    dataset = DatasetReader(input_list, target_list, None)
    train_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(srn.parameters(), lr=1e-3,
                           betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    loss_fn = torch.nn.MSELoss()

    trainer = pytt.DeepNetTrainer(
        model=srn, criterion=loss_fn, optimizer=optimizer)
    trainer.load_state(args.model_path)
    log.info(trainer.score_loader(train_loader))
