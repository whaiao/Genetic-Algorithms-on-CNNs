# coding utf-8
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim


# add seperate classes for skip connections
class SkipConnect(nn.Module):
    def __init__(self, c_in, c_out):
        super(SkipConnect, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.filter_size = (3, 3)
        self.conv_stride = (1, 1)
        self.conv = nn.Conv2d(self.c_in, self.c_out, self.filter_size,
                              self.conv_stride, 1)
        self.bn = nn.BatchNorm2d(self.c_out)
        self.relu = nn.ReLU()

    def padding(self, x):
        in_height, in_width = x.shape[2:]
        filter_height, filter_width = 3, 3
        strides = (None, 1, 1)
        #The total padding applied along the height and width is computed as:

        if (in_height % strides[1] == 0):
            pad_along_height = max(filter_height - strides[1], 0)
        else:
            pad_along_height = max(filter_height - (in_height % strides[1]), 0)
        if (in_width % strides[2] == 0):
            pad_along_width = max(filter_width - strides[2], 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides[2]), 0)

        #Finally, the padding on the top, bottom, left and right are:

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x):
        x = self.padding(x)
        res = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        # out += res
        return out


class ModelFromDecoding(nn.Module):
    def __init__(self,
                 encoding: str,
                 device: torch.device,
                 encoding_scheme: str = 'cnn'):
        super(ModelFromDecoding, self).__init__()
        self.architecture = []
        self.enc = encoding

        self.kernel_size = self.pool_stride = (2, 2)
        self._encode()
        self.softmax = nn.Softmax(0)

        self.net = nn.Sequential(*self.architecture)

    def _encode(self):
        prev = 16
        conv = SkipConnect(3, 16)
        self.architecture.append(conv)
        for m in self.enc.split('-'):
            if m == '-': continue
            m = float(m) if '.' in m else int(m)
            if isinstance(m, int):
                conv = SkipConnect(prev, m)
                prev = m
                self.architecture.append(conv)
            if isinstance(m, float):
                pool = nn.MaxPool2d(self.kernel_size, self.pool_stride,
                                    (1, 1)) if m > 0.5 else nn.AvgPool2d(
                                        self.kernel_size, self.pool_stride,
                                        (1, 1))
                self.architecture.append(pool)

    def forward(self, x):
        x = self.net(x)
        dim = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.view(-1, dim)
        fc = nn.Linear(x.shape[-1], 10)
        x = fc(x)
        return self.softmax(x)