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


class InvGradPlane(GlobalDatasetGradOptimIter):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        super(InvGradPlane, self).__init__(
            args, model, teacher, train_loader, test_loader
        )
    
    def forward_train(self):
        print(f"Preprocessing one epoch...")
        
        self.model.zero_grad()
        
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
            if (iteration+1) % 200 == 0:
                test_top1, test_ce_loss, test_feat_loss, test_weight_loss, test_feat_layer_loss = self.test()
                test_log = f"Pretrain iter {iteration} | Top-1: {test_top1:.2f}\n"
                print(test_log)
                preprocess_file.write(test_log)

        preprocess_file.close()
            
        ckpt_path = osp.join(
            self.args.output_dir,
            "finetune.pth"
        )
        torch.save(
            {'state_dict': self.model.state_dict()}, 
            ckpt_path,
        )

    def inv_train(self):
        print(f"Preprocessing one epoch...")
        self.model.eval()
        self.model.zero_grad()
        

        ce = CrossEntropyLabelSmooth(self.train_loader.dataset.num_classes, self.args.label_smoothing).to('cuda')
        featloss = torch.nn.MSELoss()
        
        dataloader_iterator = iter(self.train_loader)
        for iteration in range(1000):
            try:
                batch, label = next(dataloader_iterator)
            except:
                break
                dataloader_iterator = iter(self.train_loader)
                batch, label = next(dataloader_iterator)
            batch, label = batch.cuda(), label.cuda()
            label[label!=0] = 0

            loss, top1, ce_loss, feat_loss, linear_loss, l2sp_loss, total_loss = self.compute_loss(
                batch, label, 
                ce, featloss,
            )
            loss.backward()

        for name, module in self.model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                module.weight.grad_log = torch.abs(module.weight.grad).cpu()
        self.model.zero_grad()
        

    def process_epoch(self):
        pretrained_state_dict = self.model.state_dict()
        
        self.forward_train()
        ckpt_path = osp.join(
            self.args.output_dir,
            "finetune.pth"
        )
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt["state_dict"])
        test_top1, test_ce_loss, test_feat_loss, test_weight_loss, test_feat_layer_loss = self.test()
        print(f"Fine-tuned model | Top-1: {test_top1:.2f}")

        self.inv_train()

        self.model.zero_grad()
        self.model.load_state_dict(pretrained_state_dict)

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
        
        y, i = torch.sort(conv_weights)
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


class InvGradAvg(InvGradPlane):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        super(InvGradAvg, self).__init__(
            args, model, teacher, train_loader, test_loader
        )

    def forward_train(self):
        print(f"Preprocessing one epoch...")
        
        self.model.zero_grad()
        
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
            if (iteration+1) % 200 == 0:
                test_top1, test_ce_loss, test_feat_loss, test_weight_loss, test_feat_layer_loss = self.test()
                test_log = f"Pretrain iter {iteration} | Top-1: {test_top1:.2f}\n"
                print(test_log)
                preprocess_file.write(test_log)

        preprocess_file.close()
            
        ckpt_path = osp.join(
            self.args.output_dir,
            "finetune.pth"
        )
        torch.save(
            {'state_dict': self.model.state_dict()}, 
            ckpt_path,
        )

    def inv_train(self):
        print(f"Preprocessing one epoch...")
        self.model.eval()
        self.model.zero_grad()
        
        ce = CrossEntropyLabelSmooth(self.train_loader.dataset.num_classes, self.args.label_smoothing).to('cuda')
        featloss = torch.nn.MSELoss()

        outputs = []
        for idx, (batch, label) in enumerate(self.train_loader):
            batch, label = batch.cuda(), label.cuda()
            output = self.model(batch)
            outputs.append(output.detach().cpu())
            # break
        outputs = torch.cat(outputs)
        
        output_avg = outputs.mean(0).cuda()
        
        # dataloader_iterator = iter(self.train_loader)
        # for iteration in range(1000):
        #     try:
        #         batch, label = next(dataloader_iterator)
        #     except:
        #         break
        #         dataloader_iterator = iter(self.train_loader)
        #         batch, label = next(dataloader_iterator)
        for batch, label in self.train_loader:
            batch, label = batch.cuda(), label.cuda()
            output = self.model(batch).mean(0)

            loss = featloss(output, output_avg)
            loss.backward()

        for name, module in self.model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                module.weight.grad_log = torch.abs(module.weight.grad).cpu()
        self.model.zero_grad()
        