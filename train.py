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
from SuperResNet import SuperResNet, SuperResNetVGG16
import argparse
from Logging import log
from utils import EXTENSIONS, get_image_list

from pytorch_trainer import DeepNetTrainer
from pytorch_trainer import ModelCheckpoint, PrintCallback
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="SuperResNet", description="Train Super Resolution CNN")
    parser.add_argument('--batch_size', help='Batch size',
                        type=int, required=True)
    parser.add_argument('--epochs', help='Training epochs',
                        type=int, required=True)
    parser.add_argument('--train_input_dir', help='Input list',
                        type=str, required=True)
    parser.add_argument('--train_target_dir', help='Target list',
                        type=str, required=True)
    parser.add_argument('--eval_input_dir', help='Input list',
                        type=str, required=True)
    parser.add_argument('--eval_target_dir', help='Target list',
                        type=str, required=True)                        
    parser.add_argument('--model_path', help='Path to save the model',
                        type=str, required=True)

    args = parser.parse_args()

    log.info("Configs")
    log.info("Batch Size: {}".format(args.batch_size))
    log.info("Epochs: {}".format(args.epochs))
    log.info("Train Input List: {}".format(args.train_input_dir))
    log.info("Train Target List: {}".format(args.train_target_dir))
    log.info("Eval Input List: {}".format(args.eval_input_dir))
    log.info("Eval Target List: {}".format(args.eval_target_dir))
    log.info("Model Path: {}".format(args.model_path))

    train_input_list = get_image_list(args.train_input_dir)
    train_target_list = get_image_list(args.train_target_dir)

    eval_input_list = get_image_list(args.eval_input_dir)
    eval_target_list = get_image_list(args.eval_target_dir)

    train_set = DatasetReader(train_input_list, train_target_list, None)
    eval_set = DatasetReader(eval_input_list, eval_target_list, None)
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True)

    #srn = SuperResNet()
    srn = torch.nn.DataParallel(SuperResNetVGG16()).cuda()

    optimizer = optim.Adam(srn.parameters(), lr=1e-3,
                           betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    loss_fn = torch.nn.MSELoss()

    # Callbacks
    cb_checkpoint = ModelCheckpoint(args.model_path, reset=False, verbose=1)
    cb_print = PrintCallback()

    trainer = DeepNetTrainer(model=srn, criterion=loss_fn, optimizer=optimizer, callbacks=[
                             cb_checkpoint, cb_print], use_gpu='auto')
    trainer.fit_loader(args.epochs, train_loader, valid_data=eval_loader)

    # TrainSuperResNet(batch_size=args.batch_size, epochs=args.epochs,
    #                  input_list=args.input_list, target_list=args.target_list,
    #                  model_path=args.model_path, checkpoint_path=args.checkpoint_path)
