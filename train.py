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

from pytorch_trainer import DeepNetTrainer
from pytorch_trainer import ModelCheckpoint, PrintCallback
import os


def TrainSuperResNet(batch_size, epochs, input_list, target_list, model_path, checkpoint_path=None):
    """
    """
    # TODO: transformer
    # TODO: cuda

    use_gpu = torch.cuda.is_available()

    model = SuperResNet()
    if use_gpu:
        model = model.cuda()

    dataset = DatasetReader(input_list, target_list, None)
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True)

    optimzer = optim.Adam(model.parameters(), lr=1e-3,
                          betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    loss_fn = torch.nn.MSELoss()

    for e in range(epochs):
        for batch_idx, (input_batch, target) in enumerate(train_loader):

            if use_gpu:
                input_batch = input_batch.cuda()
                target = target.cuda()

            input_batch = Variable(input_batch.float(), requires_grad=False)
            #log.info("Input shape: {}".format(input_batch.size()))
            target = Variable(target.float(), requires_grad=False)

            Y_pred = model(input_batch)
            loss = loss_fn(Y_pred, target)

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            log.info("Epoch: {} Batch ID: {} Loss: {}".format(
                e, batch_idx, loss.data[0]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="SuperResNet", description="Train Super Resolution CNN")
    parser.add_argument('--batch_size', help='Batch size',
                        type=int, required=True)
    parser.add_argument('--epochs', help='Training epochs',
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
    log.info("Epochs: {}".format(args.epochs))
    log.info("Input List: {}".format(args.input_dir))
    log.info("Target List: {}".format(args.target_dir))
    log.info("Model Path: {}".format(args.model_path))

    input_list = get_image_list(args.input_dir)
    target_list = get_image_list(args.target_dir)

    srn = SuperResNet()
    dataset = DatasetReader(input_list, target_list, None)
    train_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(srn.parameters(), lr=1e-3,
                           betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    loss_fn = torch.nn.MSELoss()

    # Callbacks
    cb_checkpoint = ModelCheckpoint(args.model_path, reset=False, verbose=1)
    cb_print = PrintCallback()

    trainer = DeepNetTrainer(model=srn, criterion=loss_fn, optimizer=optimizer, callbacks=[
                             cb_checkpoint, cb_print], use_gpu='auto')
    trainer.fit_loader(args.epochs, train_loader, valid_data=None)

    # TrainSuperResNet(batch_size=args.batch_size, epochs=args.epochs,
    #                  input_list=args.input_list, target_list=args.target_list,
    #                  model_path=args.model_path, checkpoint_path=args.checkpoint_path)
