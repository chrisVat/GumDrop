import torch
import collections
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import cifar_pretrained_models as CIFAR_Models

import pickle
import os
import time
import argparse
import numpy as np
import json
import collections

import argparse

from network import *
import the_agent as Agent

from Utils.progress.bar import Bar
from utils import *
from gumbel_softmax import *


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="Resnet Finetuner")

parser.add_argument('-d', '--dataset', default='cifar10', type=str, help="cifar10, cifar100 or imagenet")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='Number of workers.')
parser.add_argument('--epochs', default=1, type=int, help='Total number of epochs')
parser.add_argument('--test-batch', default=1, type=int, metavar='N', help='test batchsize')
parser.add_argument('--layers', default=32, type=int, help='Layers of the Resnet (32, 110)')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='1', type=str, help='ID of GPU to use. example: 1 or 1_2 for multi gpu')
parser.add_argument('--net-load', default=None, type=str, metavar='PATH', help="Location of Net to load (if not using default)")
parser.add_argument('--agent-load', default=None, type=str, metavar='PATH', help="Location of Agent to finetune."))
parser.add_argument('--experiment', default=False, type=str2bool, help="Set to true if you are running multiple hardcoded experiments.")


class FConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(FConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FConv2d, self).forward(x)
        output_area = output.size(-1)*output.size(-2)
        filter_area = np.prod(self.kernel_size)
        if not torch.equal(x, output):
            self.num_ops += 2*self.in_channels*self.out_channels*filter_area*output_area
        else:
            self.num_ops = 0
        return output


class FLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(FLinear, self).__init__(in_features, out_features, bias)
        self.num_ops = 0

    def forward(self, x):
        output = super(FLinear, self).forward(x)
        self.num_ops += 2*self.in_features*self.out_features
        return output


def count_flops(model, reset=True):
    op_count = 0
    for m in model.modules():
        if hasattr(m, 'num_ops'):
            op_count += m.num_ops
            if reset: # count and reset to 0
                m.num_ops = 0

    return op_count



def test(epoch, val_loader, net, policy, device, criterion, layers, args):
    net.eval()

    top1 = AverageMeter()  # running average from utils, initialize them
    losses = AverageMeter()
    total_ops = []
    bar = Bar('Testing '+args.name, max=len(val_loader))

    netBlocks = sum(net.module.layer_config) if args.parallel else sum(net.layer_config)
    policyCount = torch.zeros(netBlocks, dtype=torch.int).to(device)
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):  # same as train without updating
            images, labels = Variable(images.to(device)), Variable(labels.to(device))

            probs = agent(images)

            policy = policy.repeat((len(images), 1))

            outputs = net.forward_single(images, policy)
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(labels.data).cpu().sum()
            top1.update(correct.item() * 100 / (labels.size(0) + 0.0), labels.size(0))

            loss = criterion(outputs, labels)
            losses.update(loss.item(), labels.size(0))
            ops = count_flops(net)
            total_ops.append(ops)
            
            bar.suffix = 'Epoch: {epoch}{name} | ({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    epoch=epoch+1,
                    name= args.name,
                    batch=i + 1,
                    size=len(val_loader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,)
            bar.next()
    bar.finish()

    ops_mean, ops_std = np.mean(total_ops), np.std(total_ops)
    print("FLOPs/Image: " + ops_mean + ",  FLOPs/Image Stdev = " + ops_std)
    return


def get_model(model, path, dataset=None, device=torch.device('cpu')):
    if dataset.upper() == "CIFAR100":
        num_class = 100
        source_state = torch.load(path, map_location=device)
    elif dataset.upper() == "CIFAR10":
        num_class = 10
        source_state = torch.load(path, map_location=device)
    elif dataset.upper() == "IMAGENET":
        num_class = 1000
        source_state = torchvision.models.resnet101(pretrained=True, progress=True).state_dict()
    if model == 'resnet32':
        rnet = CIFAR_Models.CifarResNet(CIFAR_Models.BasicBlock, [5, 5, 5], num_classes=num_class)
    elif model == 'resnet110':
        rnet = CIFAR_Models.CifarResNet(CIFAR_Models.BasicBlock, [18, 18, 18], num_classes=num_class)
    elif model == 'resnet101':
        rnet = CIFAR_Models.Imagenet_resnet101()

    source_state = torch.load(path, map_location=device)
    good_state_dict = collections.OrderedDict()

    if 'state_dict' in source_state:
        source_state = source_state.get('state_dict')

    for k, v in source_state.items():
        if 'module.' in k:
            k = k[7:]
        if "layer" in k:
            val = str(int(k[5]) - 1)
            k = "blocks." + val + k[6:]
        if "linear" in k:
            k = "fc" + k[6:]
        good_state_dict.update({k: v})
    # print("ALTERED LOADED: ", good_state_dict.keys())
    rnet.load_state_dict(good_state_dict, strict=True)
    # rnet.load_state_dict(torch.load(path))
    return rnet


def cleanLoad(net, path, parallel, device):
    source_state = torch.load(path, map_location=device)
    final_state = collections.OrderedDict()
    for k, v in source_state.items():
        if 'module.' in k and not parallel:
            k = k[7:]
        elif 'module.' not in k and parallel:
            k='module.'+k
        final_state.update({k: v})
    net.load_state_dict(final_state)
    return net


def main(args):
    nn.Conv2d = FConv2d
    nn.Linear = FLinear

    print(torch.cuda.is_available())
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(sci_mode=False)
    args.parallel = False
    if '_' in args.device:
        args.parallel = True
        args.device = args.device.split('_')
    mapper = map(int, args.device)
    args.device = list(mapper)

    print("device: ", args.device)
    if torch.cuda.is_available():
        device = torch.device('cuda') if args.device is None else torch.device('cuda:' + str(args.device[0]))  # apologies to people with 11+ gpus.
    else:
        device = torch.device('cpu')
    print("dataset: ", args.dataset.lower())
    normalize = None
    if args.dataset.lower() == "cifar100":
        normalize = torchvision.transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023], )
        dataset = torchvision.datasets.CIFAR100
    elif args.dataset.lower() == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        dataset = torchvision.datasets.CIFAR10
    elif args.dataset.lower() == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.dataset.lower() != "imagenet":
        testTransforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
        data_test = dataset('.', train=False, download=True, transform=testTransforms)

    else:
        testTransforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
        data_test = torchvision.datasets.ImageFolder('/mnt/creeper/grad/vattheuc/ImageNet/val', testTransforms)


    val_loader = torch.utils.data.DataLoader(data_test, batch_size=args.test_batch, shuffle=True,
                                             num_workers=args.workers)
    criterion = nn.CrossEntropyLoss()

    pretrained_model_dir = os.getcwd() + "/" + args.dataset.lower() + "models/" + args.dataset.lower() + \
                           "-resnet" + str(args.layers) + ".pth"

    net = get_model("resnet" + str(args.layers), pretrained_model_dir, dataset=args.dataset.upper())
    if args.net_load:
        if os.path.isfile(args.net_load):
            print("Loading net from memory")
            net = cleanLoad(net, args.customNetPath, args.parallel, device)
        else:
            print("Net not found at: ", args.net_load)

    if args.parallel:
        net = nn.DataParallel(net, device_ids=args.device)

    net.to(device)

    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

    #  torch.cuda.synchronize()

    start_epoch = 0

    # PUT YOUR POLICY HERE!!!
    policy = [0].to(device)
    # PUT YOUR POLICY THERE!!!!!

    test(epoch, val_loader, net, agent, device, criterion, str(args.layers), args)
    return

if __name__ == '__main__':
    startArgs = parser.parse_args()
    if startArgs.experiment:
        experiments = [
            "-d cifar10 --device 0 --layers 110 --net-load CIFAR10Trained/110/trainedAgent110acc37.270blks0.259ep45c-14.pth --name a",
            ]
        for experiment in experiments:
            main(parser.parse_args(experiment.split()))
    else:
        main(startArgs)
