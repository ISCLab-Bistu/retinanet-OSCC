import torch
import torch.nn as nn
from torchinfo import summary

from nets.resnet50 import resnet50, ResidualBlock
from nets.PyramidFeatures import PyramidFeatures
from nets.classification import Classification
import math


class retinanet(nn.Module):
    def __init__(self, backbone_in_channels: int = 64, num_layers_block: list = [3, 4, 6, 3], c3_size: int = 512,
                 c4_size: int = 1024, c5_size: int = 2048, class_in: int = 256, class_out: int = 17):
        """
        :param backbone_in_channels: the channels of input features of the first Block in resNet50
        :param num_layers_block: the num_layers of each Block in resnet50
        :param c3_size: the channel number of the output[0] from resnet50
        :param c4_size: the channel number of the output[1] from resnet50
        :param c5_size: the channel number of the output[2] from resnet50
        :param class_in: the channel of input features of classification
        :param class_out: the width of the output featurs of the last conv from classification
        """
        super(retinanet, self).__init__()
        self.backbone = resnet50(ResidualBlock, backbone_in_channels, num_layers_block)
        self.fpn = PyramidFeatures(c3_size, c4_size, c5_size)
        self.classification = Classification(channels_in=class_in, out_features=class_out)

    def forward(self, x):
        f3, f4, f5 = self.backbone(x)
        out = self.fpn(f3, f4, f5)
        out = self.classification(out)
        return out


if __name__ == "__main__":
    model = retinanet()
    summary(model, input_size=(2165, 1, 1044), depth=9)
