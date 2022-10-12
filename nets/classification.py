import torch
import torch.nn as nn
import math
from torchinfo import summary


class Classification(nn.Module):
    def __init__(self, channels_in=256, out_features=17):
        """
        :param channels_in: the channel of the features from PyramidFeature
        :param out_features: the out_width of the last conv (conv6)
        """
        super(Classification, self).__init__()
        self.conv1 = nn.Conv1d(channels_in, channels_in, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels_in, channels_in, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels_in, channels_in, kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv1d(channels_in, int(channels_in / 2), kernel_size=1, stride=1)
        self.conv5 = nn.Conv1d(int(channels_in / 2), int(channels_in / 4), kernel_size=1, stride=1)
        self.conv6 = nn.Conv1d(int(channels_in / 4), int(channels_in / 16), kernel_size=1, stride=1)

        self.fcl = nn.Linear(out_features*int(channels_in / 16), 6)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.relu(out)

        out = torch.flatten(out, start_dim=1)
        out = self.fcl(out)
        return out


