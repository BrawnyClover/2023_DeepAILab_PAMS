#!/usr/bin/python3.6  
# -*- coding: utf-8 -*-

'''
2. quant weight 구현
./model/quant_ops.py/quant_weight/__init__()
./model/quant_ops.py/quant_weight/forward()

3. quant activation 구현
./model/quant_ops.py/pams_quant_act/__init__()
./model/quant_ops.py/pams_quant_act/forward()

4. Qconv 구현
./model/quant_ops.py/QuantConv2d/forward()
'''

import collections
import math
import pdb
import random
import time
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function as F


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

def quant_max(tensor):
    """
    Returns the max value for symmetric quantization.
    """
    return torch.abs(tensor.detach()).max() + 1e-8

def TorchRound():
    """
    Apply STE to clamp function.
    """
    class identity_quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            out = torch.round(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return identity_quant().apply

class quant_weight(nn.Module):
    """
    Quantization function for quantize weight with maximum.
    """

    def __init__(self, k_bits):
        super(quant_weight, self).__init__()
        self.n = k_bits
        self.round = TorchRound()

    def forward(self, input):
        max = quant_max(input)
        s = max / (pow(2, self.n - 1) - 1)
        q_weight = s * self.round(input / s)
        del s
        return q_weight

class pams_quant_act(nn.Module):
    """
    Quantization function for quantize activation with parameterized max scale.
    """
    def __init__(self, k_bits, ema_epoch=1, decay=0.9997):
        super(pams_quant_act, self).__init__()
        # Code to Here
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float))
        self.k_bits = k_bits
        self.decay = decay
        self.ema_epoch = ema_epoch
        self.epoch = 1
        self.round = TorchRound()
        self.register_buffer('max_val', torch.ones(1))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.constant_(self.alpha, 10)

    def _ema(self, x):
        max_val = torch.mean(torch.max(torch.max(torch.max(abs(x),dim=1)[0],dim=1)[0],dim=1)[0])

        if self.epoch == 1:
            self.max_val = max_val
        else:
            self.max_val = (1.0-self.decay) * max_val + self.decay * self.max_val

    def forward(self, x):
        if self.epoch > self.ema_epoch or self.training == False:
            # f(x) = max( min(x, a), -a)
            act = torch.max(torch.min(x, self.alpha), -self.alpha)
        else:
            self._ema(x)
            act = x
            self.alpha.data = self.max_val.unsqueeze(0)

        s = self.alpha / (pow(2, self.k_bits - 1) - 1)
        q_act = self.round(act/s)*s
        return q_act

class QuantConv2d(nn.Module):
    """
    A convolution layer with quantized weight.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False,k_bits=32,):
        super(QuantConv2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)
        self.k_bits = k_bits
        self.quant_weight = quant_weight(k_bits = k_bits)
        self.output = None

        
        # self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameter(self):
        stdv = 1.0/ math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias_flag:
            nn.init.constant_(self.bias,0.0)

    def forward(self, input, order=None):
        # self.conv.weight.data = self.quant_weight(self.weight)
        # output = self.conv(input)
        return nn.functional.conv2d(input, self.quant_weight(self.weight), bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        # return output

def conv3x3(in_channels, out_channels,kernel_size=3,stride=1,padding =1,bias= True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

def quant_conv3x3(in_channels, out_channels,kernel_size=3,padding = 1,stride=1,k_bits=32,bias = True):
    return QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride,padding=padding,k_bits=k_bits,bias = bias)
