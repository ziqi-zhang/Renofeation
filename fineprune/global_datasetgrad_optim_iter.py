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


class GlobalDatasetGradOptimIter(Finetuner):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        super(GlobalDatasetGradOptimIter, self).__init__(
            args, model, teacher, train_loader, test_loader
        )
        assert (
            self.args.weight_total_ratio >= 0 and
            self.args.weight_ratio_per_prune >= 0 and 
            self.args.prune_interval >= 0 and 
            self.args.weight_init_prune_ratio >= 0 and
            self.args.weight_total_ratio >= self.args.weight_init_prune_ratio
        )
        
        self.log_path = osp.join(self.args.output_dir, "prune.log")
        self.logger = open(self.log_path, "w")
        self.init_prune()
        self.logger.close()


    def process_epoch(self):
        print(f"Processing one epoch...")
        state_dict = self.model.state_dict()
        
        self.model.zero_grad()
        for module in self.model.modules():
            if ( isinstance(module, nn.Conv2d) ):
                module.raw_weight = module.weight.clone()
        
        

        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.args.trial_lr, 
            momentum=self.args.trial_momentum, 
            weight_decay=self.args.trial_weight_decay,
        )

        ce = CrossEntropyLabelSmooth(self.train_loader.dataset.num_classes, self.args.label_smoothing).to('cuda')
        featloss = torch.nn.MSELoss()

        preprocess_path = osp.join(
            self.args.output_dir,
            "preprocess"
        )
        preprocess_file = open(preprocess_path, "w")

        dataloader_iterator = iter(self.train_loader)
        for iteration in range(self.args.trial_iteration):
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
                test_log = f"Pretrain iter {iteration} | Top-1: {test_top1:.2f}\n"
                print(test_log)
                preprocess_file.write(test_log)

        preprocess_file.close()
            
        
        for module in self.model.modules():
            if ( isinstance(module, nn.Conv2d) ):
                module.weight.grad_log = torch.abs(module.weight - module.raw_weight).cpu()

        self.model.zero_grad()
        self.model.load_state_dict(state_dict)


    def normalize_ranks(self):
        for name, module in self.model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                module.grad_log = torch.abs(module.grad)
                # shape = taylor.shape
                # taylor = taylor.view(-1)
                # taylor = taylor.type(torch.FloatTensor)
                # taylor = taylor / np.sqrt(torch.sum(taylor * taylor))
                # taylor = taylor.reshape(shape)
                # module.grad_log = taylor

    def conduct_prune(
            self,
            prune_ratio,
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
        
        y, i = torch.sort(conv_weights, descending=self.args.prune_descending)
        # thre_index = int(total * prune_ratio)
        # thre = y[thre_index]
        thre_index = int(total * prune_ratio)
        thre = y[thre_index]
        log = f"Pruning threshold: {thre:.4f}"
        self.prune_record(log)

        pruned = 0
        
        zero_flag = False
        
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                # Prune conv1 is better
                # if name == "conv1":
                #     continue
                weight_copy = module.weight.grad_log.abs().clone()
                if self.args.prune_descending:
                    mask = weight_copy.lt(thre).float()
                else:
                    mask = weight_copy.gt(thre).float()

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

    def snip_weight_prune(self, prune_ratio, iteration):
        self.prune_record(
            f"Iter {iteration} prune {prune_ratio} weights"
        )

        self.process_epoch()
        # self.normalize_ranks()
        self.conduct_prune(prune_ratio)


    def prune_record(self, log):
        print(log)
        self.logger.write(log+"\n")

    def check_param_num(self):
        model = self.model        
        total = sum([module.weight.nelement() for module in model.modules() if isinstance(module, nn.Conv2d) ])
        num = total
        for m in model.modules():
            if ( isinstance(m, nn.Conv2d) ):
                num -= int((m.weight.data == 0).sum())
        ratio = (total - num) / total
        log = f"===>Check weight: Total {total}, current {num}, prune ratio {ratio:2f}"
        self.prune_record(log)

    def init_prune(self):
        ratio = self.args.weight_init_prune_ratio
        self.prune_record(f"Init prune {ratio} weights")
        self.snip_weight_prune(ratio, 0)
        self.check_param_num()


    def final_check_param_num(self):
        self.logger = open(self.log_path, "a")
        self.check_param_num()
        self.logger.close()
        
