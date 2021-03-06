import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random
from functools import partial
from operator import itemgetter
from heapq import nsmallest
import copy

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcontrib

from torchvision import transforms
from advertorch.attacks import LinfPGDAttack

from dataset.cub200 import CUB200Data
from dataset.mit67 import MIT67Data
from dataset.stanford_dog import SDog120Data
from dataset.caltech256 import Caltech257Data
from dataset.stanford_40 import Stanford40Data
from dataset.flower102 import Flower102Data

from model.fe_resnet import resnet18_dropout, resnet50_dropout, resnet101_dropout
from model.fe_mobilenet import mbnetv2_dropout
from model.fe_resnet import feresnet18, feresnet50, feresnet101
from model.fe_mobilenet import fembnetv2

from eval_robustness import advtest, myloss
from utils import *
from fineprune.finetuner import Finetuner
from fineprune.global_datasetgrad_optim_iter import GlobalDatasetGradOptimIter


class MidDeltaW(GlobalDatasetGradOptimIter):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        super(MidDeltaW, self).__init__(
            args, model, teacher, train_loader, test_loader
        )

    def conduct_prune(
            self,
            low_ratio, ratio_interval,
        ):
        model = self.model.cpu()
        total = 0
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                    total += module.weight.data.numel()
        
        conv_weights = torch.zeros(total)
        index = 0
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                size = module.weight.data.numel()
                conv_weights[index:(index+size)] = module.weight.grad_log.view(-1).abs().clone()
                index += size
        
        y, i = torch.sort(conv_weights)
        
        low_thre_index = int(total * low_ratio)
        low_thre = y[low_thre_index]
        high_thre_index = int(total * (low_ratio+ratio_interval))
        if high_thre_index >= len(y):
            high_thre_index = len(y)-1
        high_thre = y[high_thre_index]
        log = f"Pruning threshold: {low_thre:.4f} to {high_thre:.4f}"
        self.prune_record(log)

        pruned = 0
        zero_flag = False
        
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                # Prune conv1 is better
                # if name == "conv1":
                #     continue
                weight_copy = module.weight.grad_log.abs().clone()
                mask = weight_copy.gt(high_thre).float() + weight_copy.lt(low_thre).float() 

                pruned = pruned + mask.numel() - torch.sum(mask)
                # np.random.shuffle(mask)
                module.weight.data.mul_(mask)
                if int(torch.sum(mask)) == 0:
                    zero_flag = True
                remain_ratio = int(torch.sum(mask)) / mask.numel()
                log = (f"layer {name} \t total params: {mask.numel()} \t "
                f"remaining params: {int(torch.sum(mask))}({remain_ratio:.2f})")
                self.prune_record(log)
                
        if zero_flag:
            raise RuntimeError("There exists a layer with 0 parameters left.")
        log = (f"Total conv params: {total}, Pruned conv params: {pruned}, "
        f"Pruned ratio: {pruned/total:.2f}")
        self.prune_record(log)
        self.model = model.cuda()

    def snip_weight_prune(self, low_ratio, ratio_interval):

        self.process_epoch()
        # self.normalize_ranks()
        self.conduct_prune(low_ratio, ratio_interval)
        
    def init_prune(self):
        low_ratio = self.args.weight_init_prune_ratio
        ratio_interval = self.args.weight_ratio_per_prune
        log = f"Init prune ratio {low_ratio:.2f}, interval {ratio_interval:.2f}"
        self.prune_record(log)
        self.snip_weight_prune(low_ratio, ratio_interval)
        self.check_param_num()
