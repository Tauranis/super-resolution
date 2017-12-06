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
import numpy as np
import torch
import os
import pytorch_trainer as pytt
import argparse
from SuperResNet import SuperResNet, SuperResNetVGG16
from torch.autograd import Variable
from utils import get_image_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="SuperResNet", description="Super Resolution CNN camera test")
    parser.add_argument('--model_path', help='Model path',
                        type=str, required=True)
    parser.add_argument('--input_dir', help='Input dir',
                        type=str, required=True)
    parser.add_argument('--target_dir', help='Target dir',
                        type=str, required=True)
    args = parser.parse_args()
    #srn = SuperResNet()
    srn = torch.nn.DataParallel(SuperResNetVGG16(_pretrained=False)).cuda()
    
    metrics = dict(train=dict(losses=[]), valid=dict(losses=[]))
    pytt.load_trainer_state(args.model_path, srn, metrics)
    srn.eval()

    img_list = get_image_list(args.input_dir)

    for img_path in img_list:
        print(img_path)
        frame = cv2.imread(img_path).astype(np.float64)

        # Preprocess frame
        frame /= 255.0
        frame = np.rollaxis(frame, 2, 0)
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).float()

        # Predict
        predict = pytt.predict(srn, frame_tensor)
        frame_predict = np.rollaxis(np.squeeze(predict.numpy()), 0, 3)
        
        # Normalize prediction into [0,255]
        frame_predict = (np.floor(frame_predict * 255)).astype(np.uint8)

        output_path = os.path.join(args.target_dir,"")+'.'.join(img_path.split(".")[0:-1])+"_cnn."+img_path.split(".")[-1]
        cv2.imwrite(output_path,frame_predict)
