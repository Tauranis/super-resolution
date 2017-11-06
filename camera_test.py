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
import pytorch_trainer as pytt
import argparse
from SuperResNet import SuperResNet
from torch.autograd import Variable

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="SuperResNet", description="Super Resolution CNN camera test")
    parser.add_argument('--model_path', help='Path to save the model',
                        type=str, required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)

    srn = SuperResNet()
    metrics = dict(train=dict(losses=[]), valid=dict(losses=[]))
    pytt.load_trainer_state(args.model_path, srn, metrics)

    while (True):
        ret, input_frame = cap.read()

        frame = np.copy(input_frame).astype(np.float64)

        # Preprocess frame
        frame /= 255
        frame = (frame - 0.5) * 2
        frame = np.rollaxis(frame, 2, 0)
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).float()

        # Predict
        predict = pytt.predict(srn, frame_tensor)
        frame_predict = np.rollaxis(np.squeeze(predict.numpy()), 0, 3)
        print(frame_predict[100,100,1])
        # Normalize prediction into [0,255]
        frame_predict = (((frame_predict / 2) + 0.5) * 255).astype(np.uint8)
        #print("min {} max {}".format(np.amin(frame_predict),np.amax(frame_predict)))
        #print("{} {}".format(input_frame[0,0,0],frame_predict[0,0,0]))

        cv2.imshow("Input", input_frame)
        cv2.imshow("Prediction", frame_predict)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
