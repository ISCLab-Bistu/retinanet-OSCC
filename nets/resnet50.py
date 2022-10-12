from collections import OrderedDict
import torch
import torchvision.models as models

models.resnet50()
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, center_channel, stride, down_sample=None):
        super(ResidualBlock, self).__init__()

        self.stride = stride

        self.conv1 = nn.Conv1d(in_channels, center_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(center_channel)

        self.conv2 = nn.Conv1d(center_channel, center_channel, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(center_channel)

        self.conv3 = nn.Conv1d(center_channel, center_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(center_channel * 4)
        self.relu = nn.ReLU()
        self.downSample = down_sample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.stride == 2:
            identity = self.downSample(x)
        out = out + identity
        out = self.relu(out)

        return out


class resnet50(nn.Module):
    def __init__(self, block: ResidualBlock, in_channels: int, layers: list, isFlag=False):
        """

        :param block: ResidualBlock
        :param in_channels: the channel of input features of each block
        :param layers: the num layers of each block
        """
        self.in_channels = in_channels
        super(resnet50, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.isFlag = isFlag
        self.get_trainnable()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = F.pad(out, (2, 2), 'constant', 0)
        out = self.layer1(out)
        out = self.layer2(out)
        layer2 = out

        out = self.layer3(out)
        layer3 = out

        out = self.layer4(out)
        layer4 = out

        # true: resnet50. or false: retinanet
        if self.isFlag:
            out = torch.flatten(out, start_dim=1)
            out = self.fcl(out)
            return out

        return layer2, layer3, layer4

    def get_trainnable(self):
        if self.isFlag:
            self.fcl = nn.Linear(512 * 4 * 33, 6)
        else:
            self.fcl = None

    def _make_layer(self, block: ResidualBlock, center_channel: int, blocks: int):
        """

        :param block: the residualBlock
        :param center_channel:  the channels of the middle conv of the ResidualBlock
        :param blocks: the num layers of the Block
        :return: Block
        """
        stride = 2
        layers = []
        # stride=2
        if stride == 2:
            down_sample = nn.Sequential(OrderedDict([('conv', nn.Conv1d(self.in_channels,
                                                                        center_channel * block.expansion,
                                                                        kernel_size=1, stride=stride, bias=False)),
                                                     ('batchNorm', nn.BatchNorm1d(center_channel * block.expansion))]))

            layers.append(block(self.in_channels, center_channel, stride=stride, down_sample=down_sample))
        self.in_channels = center_channel * block.expansion
        # stride=1
        for i in range(1, blocks):
            layers.append(block(self.in_channels, center_channel, stride=1))
        return nn.Sequential(*layers)


if __name__ == "__main__":
    model = resnet50(ResidualBlock, 64, [3, 4, 6, 3])
    summary(model, (122, 1, 1044), depth=4)
