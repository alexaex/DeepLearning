import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1,
                               stride=strides)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                   stride=strides)
        else:
            self.conv3 = None

        self.BN1 = nn.BatchNorm2d(num_features=out_channels)
        self.BN2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, X):
        out = F.relu(self.BN1(self.conv1(X)))
        out = self.BN2(self.conv2(out))
        if self.conv3:
            out += self.conv3(X)
        else:
            out += X
        return F.relu(out)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super().__init__()

        # ###########################################
        # #########variant of  GoogLeNet  ###########
        # ############    CONV7-64     ##############
        # 7X7 receptive field with stride=2 and padding = 3 ensures that the input size cut in half
        # #########Batch Normalization ##############
        # ############Max pooling####################
        # 3X3 max pooling with stride = 2 and padding =1.

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(7, 7), stride=stride,
                               padding=padding)
        self.BN1 = nn.BatchNorm2d(num_features=out_channels)
        self.Max_pooling = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

    def forward(self, X):
        out = F.relu(self.BN1(self.conv1(X)))
        return self.Max_pooling(out)


def ResBlock(input_channels, out_channels, num_residual, first_block=False):
    block = []
    for i in range(num_residual):
        if i == 0 and not first_block:
            block.append(
                Residual(in_channels=input_channels, out_channels=out_channels, use_1x1conv=True, strides=2))
        else:
            block.append(Residual(in_channels=out_channels, out_channels=out_channels))
    return block
