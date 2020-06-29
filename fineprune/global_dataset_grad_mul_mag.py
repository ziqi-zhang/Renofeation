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


class GlobalDatasetGradOptimMulMag(GlobalDatasetGradOptim):
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
        self.model.zero_grad()
        for module in self.model.modules():
            if ( isinstance(module, nn.Conv2d) ):
                module.raw_weight = module.weight.clone()
        
        
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.args.lr, 
            momentum=self.args.momentum, 
            weight_decay=self.args.weight_decay,
        )
        ce = CrossEntropyLabelSmooth(self.train_loader.dataset.num_classes, self.args.label_smoothing).to('cuda')
        featloss = torch.nn.MSELoss()

        preprocess_path = osp.join(
            self.args.output_dir,
            "preprocess"
        )
        preprocess_file = open(preprocess_path, "w")

        dataloader_iterator = iter(self.train_loader)
        for iteration in range(1000):
            self.model.train()
            try:
                batch, label = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(self.train_loader)
                batch, label = next(dataloader_iterator)
            batch, label = batch.cuda(), label.cuda()
            optimizer.zero_grad()

            loss, top1, ce_loss, feat_loss, linear_loss, l2sp_loss, total_loss = self.compute_loss(
                batch, label, 
                ce, featloss,
            )
            loss.backward()
            optimizer.step()
            # break
            if (iteration+1) % 300 == 0:
                test_top1, test_ce_loss, test_feat_loss, test_weight_loss, test_feat_layer_loss = self.test()
                test_log = f"Pretrain iter {iteration} | Top-1: {test_top1:.2f}"
                print(test_log)
                preprocess_file.write(test_log)
        preprocess_file.close()
            
        
        for module in self.model.modules():
            if ( isinstance(module, nn.Conv2d) ):
                module.weight.grad_log = ((module.weight - module.raw_weight)*module.raw_weight).abs().cpu()

        self.model.zero_grad()
        self.model.load_state_dict(state_dict)

