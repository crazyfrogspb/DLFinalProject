import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlfinalproject.models.spectral import SpectralNorm


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


def same_pad(in_, out_, stride, ksize):
    return int(((out_ - 1) * stride - in_ + ksize) / 2) + 1


class GatedConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 batch_norm=False, activation=torch.nn.ELU(inplace=True),
                 use_gates=True, spectral=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        if spectral:
            self.conv2d = SpectralNorm(self.conv2d)
            self.mask_conv2d = SpectralNorm(self.mask_conv2d)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.use_gates = use_gates

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)

        if self.use_gates:
            if self.activation is not None:
                x = self.activation(x) * self.gated(mask)
            else:
                x = x * self.gated(mask)

        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 batch_norm=False, activation=torch.nn.ELU(inplace=True)):
        super().__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = self.conv2d(input)

        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class GatedDeconv2d(torch.nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=False,
                 activation=torch.nn.ELU(inplace=True)):
        super().__init__()
        self.conv2d = GatedConv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)


class SpectralConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation=torch.nn.ELU(inplace=True), spectral=True,
                 batch_norm=False):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        if spectral:
            self.conv2d = SpectralNorm(self.conv2d)
        self.activation = activation
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.batch_norm = batch_norm

    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            x = self.activation(x)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x
