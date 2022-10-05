import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time


# get a gumbel distribution in a tensor the same size as yours.
def sample_gumbel(shape, device, eps=1e-20):
    U = torch.FloatTensor(shape).uniform_().to(device)
    return -Variable(torch.log(-torch.log(U + eps) + eps))  # why add epsilon ?


# returns your matrix + the gumbel sample.
def gumbel_softmax_sample(logits, temperature, device):
    y = logits + sample_gumbel(logits.size(), device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax_sample_mask(logits, temperature, device, mask):
    y = logits + sample_gumbel(logits.size(), device)
    z = y.clone()
    z[:, mask == False] = -10e10
    y = (z-y).detach() + y
    return F.softmax(y / temperature, dim=-1)


# new and improved
def gumbel_softmax(logits, block_usage, device, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    #  vals = logits[:,:,1] - logits[:,:,0]
    #  y = gumbel_softmax_sample(vals, temperature)  # get the values with gumbel distribution added.
    y = gumbel_softmax_sample(logits, temperature, device)  # SOLOS
    ind = torch.zeros_like(y)
    ind = ind.scatter_(1,(torch.topk(y,block_usage, dim=1))[1],1)
    return (ind-y).detach() + y


# new and improved
def gumbel_softmax_masked(logits, block_usage, device, temperature, mask):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample_mask(logits, temperature, device, mask)
    ind = torch.zeros_like(y)
    ind = ind.scatter_(1,(torch.topk(y, block_usage, dim=1))[1],1)
    return (ind-y).detach() + y, y


# new and improved
def gumbel_softmax_epsilon(logits, block_usage, device, temperature, epsilon):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    epsilonVals = torch.zeros_like(logits).uniform_(0,1).to(device)
    maxVals = logits.max(dim=1)[0]  # max for each logits. 
    filler = torch.ones_like(logits).to(device)
    filler = filler*maxVals[:, None]
    logits[epsilonVals<epsilon] = filler[epsilonVals<epsilon]
    y = gumbel_softmax_sample(logits, temperature, device)  # SOLOS
    ind = torch.zeros_like(y)
    ind = ind.scatter_(1,(torch.topk(y,block_usage, dim=1))[1],1)
    return (ind-y).detach() + y

'''


# get a gumbel distribution in a tensor the same size as yours.
def sample_gumbel(shape, rank, eps=1e-20):
    U = torch.FloatTensor(shape).uniform_().cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps)).cuda()  # why add epsilon ?


# returns your matrix + the gumbel sample.
def gumbel_softmax_sample(logits, temperature, rank):
    torch.cuda.set_device(rank)
    y = logits + sample_gumbel(logits.size(), rank).cuda()
    return F.softmax( (y / temperature).cuda(), dim=-1).cuda()



# new and improved
def gumbel_softmax(logits, block_usage, temperature=5, rank=0):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    #  vals = logits[:,:,1] - logits[:,:,0]
    #  y = gumbel_softmax_sample(vals, temperature)  # get the values with gumbel distribution added.
    
    y = gumbel_softmax_sample(logits, temperature, rank=rank)  # SOLOS
    ind = torch.zeros_like(y)
    ind = ind.scatter_(1,(torch.topk(y,block_usage, dim=1))[1],1)
    return ((ind-y).detach() + y)
    
'''
