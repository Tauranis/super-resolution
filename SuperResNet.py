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
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules import padding
import numpy as np
from torchvision.models import vgg

from Logging import log


class SuperResNet(torch.nn.Module):
    """
    Super Resolution Network
    """

    def __init__(self):
        """
        Create conv layers
        """
        super(SuperResNet, self).__init__()

        self.conv1_3x3 = torch.nn.Conv2d(3, 10, kernel_size=3)
        self.conv2_3x3 = torch.nn.Conv2d(10, 10, kernel_size=3)
        self.conv3_1x1 = torch.nn.Conv2d(10, 3, kernel_size=1)
        self.dropout = torch.nn.Dropout2d(0.3)

        self.padding_3x3 = torch.nn.ZeroPad2d(1)

    def forward(self, input):
        """
        Perform network forwarding

        OBS: padding is applied on 3x3 convolutions only.
        """

        x = F.relu(self.conv1_3x3(self.padding_3x3(input)))
        x = F.relu(self.conv2_3x3(self.padding_3x3(x)))
        x = F.relu(self.conv3_1x1(x))

        x = self.dropout(x)
    
        return x

class SuperResNetVGG16(torch.nn.Module):
    """
    Super Resolution Network
    """

    def __init__(self,_pretrained = True):
        """
        Create conv layers
        """
        super(SuperResNetVGG16, self).__init__()

        vgg_full = vgg.vgg16(pretrained=_pretrained)
        self.vgg_conv = torch.nn.Sequential(vgg_full.features[0],vgg_full.features[1],vgg_full.features[2],vgg_full.features[3])
        self.conv3_1x1 = torch.nn.Conv2d(64, 3, kernel_size=1)
        del vgg_full


        #self.padding_3x3 = torch.nn.ZeroPad2d(1)

    def forward(self, input):
        """
        Perform network forwarding

        OBS: padding is applied on 3x3 convolutions only.
        """   
        x = self.vgg_conv(input)
        x = torch.clamp(self.conv3_1x1(x),0,1)
        
        return x


def main():

    #srn = SuperResNet()
    srn = SuperResNetVGG16()
    # OBS: Pytorch is channels first (batch_size,n_channels,width,height)
    input_sample = Variable(torch.from_numpy(
        np.random.rand(1, 3, 300, 300)), requires_grad=False).double()
    output = srn.double().forward(input_sample)
    log.info(output.size())


if __name__ == "__main__":
    main()
