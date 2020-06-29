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
from fineprune.global_datasetgrad_optim import GlobalDatasetGradOptim


class GlobalDatasetGradOptimDivMag(GlobalDatasetGradOptim):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        super(GlobalDatasetGradOptim, self).__init__(
            args, model, teacher, train_loader, test_loader
        )

    def process_epoch(self):
        print(f"Processing one epoch...")
        state_dict = self.model.state_dict()
        self.model.eval()
        self.model.zero_grad()
        for module in self.model.modules():
            if ( isinstance(module, nn.Conv2d) ):
                module.raw_weight = module.weight.clone()
        
        
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=5e-3, 
            momentum=0.9, 
            weight_decay=0,
        )
        ce = CrossEntropyLabelSmooth(self.train_loader.dataset.num_classes, self.args.label_smoothing).to('cuda')
        featloss = torch.nn.MSELoss()

        for batch, label in self.train_loader:
            batch, label = batch.cuda(), label.cuda()
            optimizer.zero_grad()

            loss, top1, ce_loss, feat_loss, linear_loss, l2sp_loss, total_loss = self.compute_loss(
                batch, label, 
                ce, featloss,
            )
            loss.backward()
            optimizer.step()
            # break
        
        for module in self.model.modules():
            if ( isinstance(module, nn.Conv2d) ):
                weight_diff = (module.weight - module.raw_weight).abs()
                raw_weight_abs = module.raw_weight.clone().abs()
                raw_weight_abs[raw_weight_abs < self.args.weight_low_bound] = self.args.weight_low_bound
                module.weight.grad_log = (weight_diff / raw_weight_abs).cpu()

        self.model.zero_grad()
        self.model.load_state_dict(state_dict)

