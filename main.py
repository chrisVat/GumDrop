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

parser.add_argument('--make-agent', default=True, type=str2bool, help='True for making an agent, False for finetuning agent and net.')
parser.add_argument('--epochs', default=300, type=int, help='Total number of epochs')
parser.add_argument('--train-batch', default=256, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test-batch', default=512, type=int, metavar='N', help='test batchsize')
parser.add_argument('--agent-lr', default=0.1, type=float, metavar='LR', help='initial learning rate of agent')
parser.add_argument('--net-lr', default=0.1, type=float, metavar='LR', help='initial learning rate of net')
parser.add_argument('--step-size', default=70, type=int, metavar='N', help='Steps until LR decay.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR Multiplier every step-size epochs.')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 0)')
parser.add_argument('--layers', default=32, type=int, help='Layers of the Resnet (32, 110)')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--temp', default=50, type=int, metavar='N', help='Temperature of gumbel softmax.')
parser.add_argument('--device', default=0, type=int, help='ID of GPU to use.')
parser.add_argument('--block-usage', default=0.47, type=float, help='Percentage of Network the Agent can use.')
parser.add_argument('--net-load', default=None, type=str, metavar='PATH', help="Location of Net to load (if not using default)")
parser.add_argument('--agent-load', default=None, type=str, metavar='PATH', help="Location of Agent to finetune.")
parser.add_argument('--reload', default=True, type=str2bool, help="Reload best Agent/Net on new LR.")
parser.add_argument('--printTestPolicy', default=False, type=str2bool, help="Print the layer usage on validation set.")
parser.add_argument('--printTrainPolicy', default=False, type=str2bool, help="Print the layer usage on train set.")
parser.add_argument('--printRemoved', default=False, type=str2bool, help="Print which layers have been removed each epoch.")
parser.add_argument('--printTrainProbs', default=False, type=str2bool, help="Print the sum of agent probs from train set.")
parser.add_argument('--agent-save-location', default="trainedAgent", type=str, help="Where to save trained Agents.")
parser.add_argument('--net-save-location', default="trainedNet", type=str, help="Where to save trained Nets.")
parser.add_argument('--experiment-name', default='', type=str, help="Used for logging different experiments with hyperparameters.")
parser.add_argument('--experiment', default=False, type=str2bool, help="Set to true if you are running multiple hardcoded experiments.")


def train(epoch, train_loader, net, agent, agent_optimizer, device, netOptimizer, criterion, items, args):
    if args.make_agent:
        net.eval()
    else:
        net.train()
    agent.train()  # just changes mode, train or eval.

    top1 = AverageMeter()  # running average from utils, initialize them
    losses = AverageMeter()

    netBlocks = sum(net.layer_config)
    policyCount = torch.zeros(netBlocks, dtype=torch.int).to(device)
    bar = Bar('Training ' + args.experiment_name, max=len(train_loader))

    for i, task_batch in enumerate(train_loader):
        images = task_batch[0]
        labels = task_batch[1]

        images, labels = Variable(images.to(device)), Variable(labels.to(device))
        probs = agent(images)
        policy, yProbs = gumbel_softmax_masked(probs.view(probs.size(0), -1), int(netBlocks * args.block_usage), device,
                                               args.temp, args.notRemoved)  # SOLOS

        outputs = net.forward(images, policy)

        if args.printTrainPolicy:
            policyCount += policy.sum(0, dtype=torch.int)

        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(labels.data).sum()
        top1.update(correct.item() * 100 / (labels.size(0) + 0.0), labels.size(0))

        # Loss
        loss = criterion(outputs, labels)  # criterion = nn.CrossEntropyLoss()
        losses.update(loss.item(), labels.size(0))

        #  tasks_losses.update(loss.item(), labels.size(0))  # running statistics
        agent_optimizer.zero_grad()
        netOptimizer.zero_grad()
        loss.backward()
        agent_optimizer.step()

        if not args.make_agent:
            netOptimizer.step()

        bar.suffix = 'Epoch: {epoch}{name} | ({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    epoch=epoch+1,
                    name= args.experiment_name,
                    batch=i + 1,
                    size=len(train_loader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,)
        bar.next()
    bar.finish()
    if args.printTrainPolicy:
        print("TRAINING Policy sums: ", policyCount)
    if args.printTrainProbs:
        print("Training Policy probs[0]: ", yProbs[0])
    return

def test(epoch, val_loader, net, agent, device, criterion, layers, args):
    net.eval()
    agent.eval()  # change to eval mode.

    top1 = AverageMeter()  # running average from utils, initialize them
    losses = AverageMeter()

    bar = Bar('Testing '+args.experiment_name, max=len(val_loader))

    netBlocks = sum(net.layer_config)
    if not args.make_agent:
        policyCount = torch.zeros(netBlocks, dtype=torch.int).to(device)
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):  # same as train without updating
            images, labels = Variable(images.to(device)), Variable(labels.to(device))

            probs = agent(images)

            if args.notRemoved is not None:
                probs[:, args.notRemoved == False] = -10e10  # it doesn't let me put not args.notRemoved
            policy = torch.zeros_like(probs)
            policy = policy.scatter_(1, (torch.topk(probs, int(args.block_usage * netBlocks), dim=1))[1], 1)

            if not args.make_agent:
                policyCount += policy.sum(0, dtype=torch.int)

            outputs = net.forward(images, policy)
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(labels.data).cpu().sum()
            top1.update(correct.item() * 100 / (labels.size(0) + 0.0), labels.size(0))

            loss = criterion(outputs, labels)
            losses.update(loss.item(), labels.size(0))
            bar.suffix = 'Epoch: {epoch}{name} | ({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    epoch=epoch+1,
                    name= args.experiment_name,
                    batch=i + 1,
                    size=len(val_loader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,)
            bar.next()
    bar.finish()

    if args.make_agent and top1.avg > args.currentBest:
        print("Saving this model. It's good.")
        args.bestModelName = layers + "acc" + format(top1.avg, '.3f') + "blks" + \
                             format(policy.sum() / (policy.size(0) * policy.size(1)), '.3f') + \
                             "ep" + str(epoch) + args.experiment_name + ".pth"
        agent_state_dict = agent.state_dict()
        torch.save(agent_state_dict, (os.getcwd() + '/' + args.agent_save_location + "/trainedAgent" + args.bestModelName))
        args.currentBest = top1.avg

    if not args.make_agent and top1.avg > args.currentBest:
        print("Saving this model. It's good.")

        args.bestModelName = layers + "acc" + format(top1.avg, '.3f') + "blks" + \
                             format(policy.sum() / (policy.size(0) * policy.size(1)), '.3f') + \
                             "ep" + str(epoch) + args.experiment_name + ".pth"

        net_state_dict = net.state_dict()
        agent_state_dict = agent.state_dict()
        torch.save(agent_state_dict, (os.getcwd() + '/' + args.agent_save_location + "/trainedAgent" + args.bestModelName))
        torch.save(net_state_dict, (os.getcwd() + '/' + args.net_save_location + "/trainedNet" + args.bestModelName))

        args.blockUsageSum = policyCount.clone()
        args.currentBest = top1.avg

    if args.printTestPolicy:
        print("Policy sums: ", policyCount)
        policyScore = 0
        unused = 0
        for i in policyCount:
            if i != 0 and i != len(val_loader.dataset):
                policyScore += 1
            elif i == 0:
                unused += 1
        print("Policy Score: ", policyScore, "/", netBlocks, ".  Unused Blocks: ", unused)
    return


def get_model(model, path, dataset=None, device=torch.device('cpu')):
    if dataset.upper() == "CIFAR100":
        num_class = 100
    elif dataset.upper() == "CIFAR10": 
        num_class = 10
    if model == 'resnet20':
        rnet = CIFAR_Models.CifarResNet(CIFAR_Models.BasicBlock, [3, 3, 3], num_classes=num_class)
    if model == 'resnet32':
        rnet = CIFAR_Models.CifarResNet(CIFAR_Models.BasicBlock, [5, 5, 5], num_classes=num_class)
    if model == 'resnet44':
        rnet = CIFAR_Models.CifarResNet(CIFAR_Models.BasicBlock, [7, 7, 7], num_classes=num_class)
    if model == 'resnet56':
        rnet = CIFAR_Models.CifarResNet(CIFAR_Models.BasicBlock, [9, 9, 9], num_classes=num_class)
    if model == 'resnet110':
        rnet = CIFAR_Models.CifarResNet(CIFAR_Models.BasicBlock, [18, 18, 18], num_classes=num_class)
    # print("LOADED: ", torch.load(path).keys())
    # print("DESTINATION: ", rnet.state_dict().keys())
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


def parallel_load(agent, path):
    agent_state_dict = collections.OrderedDict()
    for k, v in torch.load(path).items():
        k = "module." + k
        agent_state_dict.update({k: v})
    agent.load_state_dict(agent_state_dict, strict=True)
    return agent


def main(args):
    args.blockUsageSum = None
    args.notRemoved = None
    args.removeCount = None
    args.bestModelName = None
    args.currentBest = 0
    print("EXPERIMENT NAME: ", args.experiment_name) 
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(sci_mode=False)
    if torch.cuda.is_available():
        device = torch.device('cuda') if args.device is None else torch.device('cuda:' + str(args.device))
    else:
        device = torch.device('cpu')

    print("dataset: ", args.dataset.lower())
    normalize = None
    if args.dataset.lower() == "cifar100":
        normalize = torchvision.transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023], )
        dataset = torchvision.datasets.CIFAR100
    elif args.dataset.lower() == "cifar10":
        noramlize = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dataset = torchvision.datasets.CIFAR10

    trainTransforms = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(size=32, padding=4),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.ToTensor(), normalize, ])
    testTransforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])

    data_train = dataset('.', train=True, download=True, transform=trainTransforms)
    data_test = dataset('.', train=False, download=True, transform=testTransforms)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.train_batch, shuffle=True,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(data_test, batch_size=args.test_batch, shuffle=True,
                                             num_workers=args.workers)
    criterion = nn.CrossEntropyLoss()

    pretrained_model_dir = os.getcwd() + "/" + args.dataset.lower() + "models/" + args.dataset.lower() + \
                           "-resnet" + str(args.layers) + ".pth"

    net = get_model("resnet" + str(args.layers), pretrained_model_dir, dataset=args.dataset.upper())
    if args.net_load:
        if os.path.isfile(args.net_load):
            print("Loading net from memory")
            net.load_state_dict(torch.load(args.customNetPath + ".pth"), strict=True, map_location=device)
        else:
            print("Net not found at: ", args.net_load)

    agent = Agent.resnet(sum(net.layer_config))  # SOLOS
    if args.agent_load:
        if os.path.isfile(args.agent_load):
            print("Loading agent from memory.")
            print(args.agent_load)
            agent.load_state_dict(torch.load(args.agent_load, map_location=device))
        else:
            print("Agent not found at: ", args.agent_load)

    net.to(device)
    agent.to(device)

    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

    optimizer = optim.Adam(agent.parameters(), lr=args.agent_lr, weight_decay=args.weight_decay)  # add momentum?
    netOptimizer = optim.Adam(net.parameters(), lr=args.net_lr )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)  # changed r
    netScheduler = torch.optim.lr_scheduler.StepLR(netOptimizer, step_size=args.step_size, gamma=args.gamma)

    #  torch.cuda.synchronize()

    start_epoch = 0
    netBlocks = sum(net.layer_config)
    if args.removeCount is None:
        args.removeIncrement = (netBlocks - int(args.block_usage * netBlocks)) / (max(args.epochs // args.step_size,1))
        if (netBlocks - int(args.block_usage * netBlocks)) % (args.epochs / args.step_size) == 0:
            args.removeIncrement = (netBlocks - int(args.block_usage * netBlocks)) / (
                        max((args.epochs // args.step_size) - 1, 1))
        if args.step_size == 1:
            args.removeIncrement = (netBlocks - int(args.block_usage * netBlocks)) / (args.epochs - 1)
        args.removeCount = args.removeIncrement + 0.01  # the 0.01 prevents weird floating point number problems
        args.notRemoved = torch.tensor(True, dtype=torch.bool).expand(netBlocks).to(device)
        args.blockUsageSum = torch.zeros_like(args.notRemoved, dtype=torch.float).to(device)
    if not args.make_agent:
        print(args.removeIncrement, " blocks will be removed every ", args.step_size, " epochs. ")   
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if epoch % args.step_size == 0 and args.reload and args.bestModelName is not None:
            agent.load_state_dict(torch.load(os.getcwd() + '/' + args.agent_save_location + "/trainedAgent" + args.bestModelName))
            if not args.make_agent:
                net.load_state_dict(torch.load(os.getcwd() + '/' + args.net_save_location + "/trainedNet" + args.bestModelName))
            print("Loading best performing models: trainedAgent/Net", args.bestModelName)
        if not args.make_agent and epoch % args.step_size == 0 and epoch > 0:
            args.blockUsageSum[args.notRemoved == False] = args.blockUsageSum.max() + 1
            args.notRemoved[torch.topk(args.blockUsageSum, int(args.removeCount), largest=False)[1]] = False  # remove the least used (remaining) layers
            args.removeCount = args.removeCount - int(args.removeCount) + args.removeIncrement  # if size change isn't divisble by steps.
        train(epoch, train_loader, net, agent, optimizer, device, netOptimizer, criterion, len(data_train), args)
        test(epoch, val_loader, net, agent, device, criterion, str(args.layers), args)
        if not args.make_agent:
            netScheduler.step()
        scheduler.step()
        if args.printRemoved:
            print(args.notRemoved)
    # args.blockUsageSum[args.notRemoved == True] = 0
    return args.currentBest

if __name__ == '__main__':
    startArgs = parser.parse_args()
    if startArgs.experiment:
        experiments = [
            "-d cifar10 --make-agent True --layers 110 --epochs 50 --agent-lr 1e-4 --step-size 50 --experiment-name a",
            "-d cifar10 --make-agent True --layers 110 --epochs 50 --agent-lr 3e-4 --step-size 50 --experiment-name b",
            "-d cifar10 --make-agent True --layers 110 --epochs 50 --agent-lr 5e-4 --step-size 50 --experiment-name c",
            "-d cifar10 --make-agent True --layers 110 --epochs 50 --agent-lr 7e-4 --step-size 50 --experiment-name d",
            "-d cifar10 --make-agent True --layers 110 --epochs 50 --agent-lr 9e-4 --step-size 50 --experiment-name e",
            ]
        f = open("log"+str(startArgs.device)+".txt", "w")
        f.write("LOG\n\n")
        for experiment in experiments:
            f.write(experiment + "\n\n")
            bestAccuracy = main(parser.parse_args(experiment.split()))
            f.write("Best accuracy: " + str(bestAccuracy) + "\n\n")
        f.close()
    else:
        main(startArgs)
