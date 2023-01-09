import torch
from torch import nn


# [224-11+2]/4+1

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(11, 11), padding=(1, 1), stride=4), nn.ReLU(),
            # [1,1,224,224] -> [1,96,54,54]
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # [1,96,54,54] -> [1,96,26,26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=(2, 2)), nn.ReLU(),
            # [1,96,26,26] -> [1,256,26,26]
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # [1,256,26,26] -> [1,256,12,12]
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(),
            # [1,256,12,12] -> [1,384,12,12]
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(),
            # [1,384,12,12] -> [1,384,12,12]
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=(1, 1)), nn.ReLU(),
            # [1,384,12,12] -> [1,256,12,12]
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
            # [1,256,12,12] -> [1,256,5,5]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # [1,256,5,5] -> [1,256*5*5]
            # Dense
            nn.Linear(256 * 5 * 5, 256 * 5 * 5), nn.ReLU(), nn.Dropout(0.5),
            # [1,256*5*5] -> [1,256*5*5]
            nn.Linear(256 * 5 * 5, 256 * 5 * 5), nn.ReLU(), nn.Dropout(0.5),
            # [1,256*5*5] -> [1,256*5*5]
            nn.Linear(256 * 5 * 5, 10)
            # [1,256*5*5] -> [1,10]
        )

    def forward(self, In):
        return self.classifier(self.feature(In))

