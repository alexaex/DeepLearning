import torch
from torch import nn
import torchvision
from torch.nn import functional as F
import torchinfo


class Inception(nn.Module):
    def __init__(self, in_channels, path1, path2, path3, path4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1 = nn.Conv2d(in_channels=in_channels, out_channels=path1, kernel_size=(1, 1))

        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=path2[0], kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=path2[0], out_channels=path2[1], kernel_size=(3, 3), padding=(1, 1))
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=path3[0], kernel_size=(1, 1)), nn.ReLU(),
            nn.Conv2d(in_channels=path3[0], out_channels=path3[1], kernel_size=(5, 5), padding=(2, 2))
        )

        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=path4, kernel_size=(1, 1))
        )

    def forward(self, x):
        path1 = F.relu(self.p1(x))
        path2 = F.relu(self.p2(x))
        path3 = F.relu(self.p3(x))
        path4 = F.relu(self.p4(x))
        return torch.cat((path1, path2, path3, path4), dim=1)

