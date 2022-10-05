#  This is the model that agent acts on.
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride=2):
        super(DownsampleB, self).__init__()
        self.avg = nn.AvgPool2d(stride)  # makes H and W smaller

    def forward(self, x):
        residual = self.avg(x)
        return torch.cat((residual, residual * 0), 1)  # pads with zeros for projection shortcut


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# No projection: identity shortcut
class BasicBlock(nn.Module):  # standard
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Sequential(nn.ReLU(True), conv3x3(planes, planes))
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        y = self.bn2(out)

        return y


class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=10):  # layers means blocks per layer.
        super(ResNet, self).__init__()

        factor = 1
        self.in_planes = int(32 * factor)
        self.conv1 = conv3x3(3, int(32 * factor))  # first layer 3 channels.
        self.bn1 = nn.BatchNorm2d(int(32 * factor))
        self.relu = nn.ReLU(inplace=True)

        strides = [2, 2, 2]  # strides for the downsampling.
        filt_sizes = [64, 128, 256]  # 64 depth, then 128 then 256
        self.blocks, self.ds = [], []

        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):  # create main blocks.
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)
        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)

        self.bn2 = nn.Sequential(nn.BatchNorm2d(int(256 * factor)), nn.ReLU(True))
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # we are just essentially resizing it i think smart.
        self.linear = nn.Linear(int(256 * factor), num_class)

        self.layer_config = layers
        for m in self.modules():  # initialize values for everything.
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.bn1(self.conv1(x))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.in_planes != planes * block.expansion:  # more out layers than in?
            downsample = DownsampleB(self.in_planes, planes * block.expansion, 2)  # downsample it.

        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion  # reset the input planes.
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))  # add the rest of the blocks.

        return layers, downsample

    def forward(self, x, policy=None):  # go forward with the policy given by the agent.
        t = 0  # Changed to work to delete blocks instead
        x = self.seed(x)

        if policy is not None:
            for segment, num_blocks in enumerate(self.layer_config):  # segment just i=0,i++, num_blocks is # per layer.
                for b in range(num_blocks):  # for each block
                    action = policy[:, t].contiguous()  # makes the elements contiguous (exist as if it were made)
                    # as it is not have its meta information changed. Not having this might cause errors.
                    # from this we get 1's and zeros, as to whether to use fine tune or not.
                    # here we are just getting the t'th element, so what to do for this given layer.
                    action_mask = action.float().view(-1, 1, 1, 1)  # 1 = use frozen, 0 = finetune.
                    # .float converts to float... it already is one. action_mask: tensor([[[[0.]]]], device='cuda:0')

                    residual = self.ds[segment](x) if b == 0 else x  # downsample on first of layer.
                    output = self.blocks[segment][b](x)  # run the block on x.

                    f1 = F.relu(residual + output)  # Using the block
                    x = f1 * (action_mask) + residual*(1-action_mask)  # decide which to use.
                    t += 1  # go for each layer
        else:
            for segment, num_blocks in enumerate(self.layer_config):  # no policy
                for b in range(num_blocks):
                    residual = self.ds[segment](x) if b == 0 else x  # fine tune layers
                    output = self.blocks[segment][b](x)
                    x = F.relu(residual + output)
                    t += 1
        x = self.bn2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def resnet26(num_class=10, blocks=BasicBlock):
    return ResNet(blocks, [4, 4, 4], num_class)
