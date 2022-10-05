# ResNet for determining which layers to fine tune.
# Number of classes = number of blocks in net*2

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride=2):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)  # make smaller.

    def forward(self, x):
        residual = self.avg(x)
        return torch.cat((residual, residual*0), 1)  # doubles length of each row


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# They are making a BasicBlock that won't be using projection, only the identity shortcut.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Sequential(nn.ReLU(True), conv3x3(planes, planes))  # means inPlace = True, saves memory.
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)  # they had y = self.bn2(out) before but I didn't see the point.

        return out  # changed to returning out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=10):
        super(ResNet, self).__init__()

        factor = 1
        self.in_planes = int(32*factor)
        self.conv1 = conv3x3(3, int(32*factor))  # 3 input channels (colours), 32*factor output channels.
        self.bn1 = nn.BatchNorm2d(int(32*factor))
        self.relu = nn.ReLU(inplace=True)

        strides = [2, 2, 2]
        filt_sizes = [64, 128, 256]
        self.blocks, self.ds = [], []  # Downsample
        # removed parallel blocks

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)  # create list of blocks
        self.ds = nn.ModuleList(self.ds)  # create list of downsampling that will happen

        self.bn2 = nn.Sequential(nn.BatchNorm2d(int(256*factor)), nn.ReLU(True))  # final layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 1x1 avgPool
        self.linear = nn.Linear(int(256*factor), num_class)  # final FC layer.

        self.layer_config = layers  # number of blocks in each layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # is m is Conv2d
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # number of weights
                m.weight.data.normal_(9, math.sqrt(2./n))  # initializes the weights for the filters.
            elif isinstance(m, nn.BatchNorm2d):  # if m is BatchNorm2d
                m.weight.data.fill_(1)  # initialize weights for batch norm
                m.bias.data.zero_()  # initialize weights for batchnorm.

    def seed(self, x):  # returns x after a convolution and batch norm.
        x = self.bn1(self.conv1(x))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential()  # do nothing
        if stride != 1 or self.in_planes != planes * block.expansion:  # if the size changes
            downsample = DownsampleB(self.in_planes, planes*block.expansion, 2)   # set the downsample.

        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return layers, downsample

    def forward(self, x):
        t = 0
        x = self.seed(x)

        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b==0 else x
                output = self.blocks[segment][b](x)
                x = F.relu(residual+output)
                t += 1
        x = self.bn2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def resnet(num_class=10, blocks=BasicBlock):
    return ResNet(blocks, [1,1,1], num_class)

