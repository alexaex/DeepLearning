import torch
from torch import nn


def VGG_Block(Conv_nums, input_channels, out_putchannels):
    layers = []
    for _ in range(Conv_nums):
        layers.append(nn.Conv2d(input_channels, out_putchannels, kernel_size=(3, 3), padding=1))
        input_channels = out_putchannels
        # [num_batches,channels,h,w]->[num_batches,out_channels,h,w]
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
    return nn.Sequential(*layers)


def VGG_Net(conv_arch):
    conv_blks = []
    _input_channels = 1
    for (conv_nums, out_channels) in conv_arch:
        conv_blks.append(VGG_Block(conv_nums, _input_channels, out_channels))
        _input_channels = out_channels
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )
