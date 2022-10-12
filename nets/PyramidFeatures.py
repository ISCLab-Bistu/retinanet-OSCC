import torch.nn.functional as F
import torch.nn as nn
from torchinfo import summary


class PyramidFeatures(nn.Module):
    def __init__(self, c3_size: int, c4_size: int, c5_size: int, feature_size: int = 256):
        """

        :param c3_size: the channels of the output[0] of resnet50
        :param c4_size: the channels of the output[1] of resnet50
        :param c5_size: the channels of the output[2] of resnet50
        :param feature_size: the channels of each output features
        """
        super(PyramidFeatures, self).__init__()
        self.f5_1 = nn.Conv1d(c5_size, feature_size, kernel_size=1, stride=1)

        self.f4_1 = nn.Conv1d(c4_size, feature_size, kernel_size=1, stride=1)

        self.f3_1 = nn.Conv1d(c3_size, feature_size, kernel_size=1, stride=1)
        self.f3_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x3, x4, x5):
        """

        :param x3: the 3rd layer output of resnet50
        :param x4: the 4th layer output of resnet50
        :param x5: the 5th layer output of resnet50
        :return: the merge features
        """
        _, _, w3 = x3.size()
        _, _, w4 = x4.size()

        out_5 = self.f5_1(x5)
        out_5_upsampled = F.interpolate(out_5, size=w4)

        out_4 = self.f4_1(x4)
        out_4 = out_5_upsampled + out_4
        out_4_upsampled = F.interpolate(out_4, size=w3)

        out_3 = self.f3_1(x3)
        out_3 = out_3 + out_4_upsampled
        out_3 = self.f3_2(out_3)
        out_3 = self.relu(out_3)
        return out_3


if __name__ == "__main__":
    model = PyramidFeatures(512, 1024, 2048)
    summary(model, input_size=((35, 512, 132), (35, 1024, 66), (35, 2048, 33)))
