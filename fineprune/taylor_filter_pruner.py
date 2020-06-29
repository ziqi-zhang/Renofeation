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

class TaylorFilterPruner(Finetuner):
    def __init__(
        self,
        args,
        model,
        teacher,
        train_loader,
        test_loader,
    ):
        super(TaylorFilterPruner, self).__init__(
            args, model, teacher, train_loader, test_loader
        )
        assert (
            self.args.filter_total_number >= 0 and
            self.args.filter_number_per_prune >= 0 and 
            self.args.prune_interval >= 0 and 
            self.args.filter_init_prune_number >= 0 and
            self.args.filter_total_number >= self.args.filter_init_prune_number
        )
        
        self.log_path = osp.join(self.args.output_dir, "prune.log")
        self.logger = open(self.log_path, "w")
        self.cal_prunable_filters()
        self.init_prune()
        self.logger.close()

    def cal_prunable_filters(self):
        filters = 0
        self.prunable_filters = {}
        self.prune_record("Cal prunable filter")
        for name, module in self.model.named_modules():
            if (
                isinstance(module, nn.Conv2d) and 
                "layer" in name and "conv1" in name
            ):
                self.prune_record(f"{name}, {module.out_channels}")
                filters += module.out_channels
                self.prunable_filters[name] = module
                module.prune_ref = True

        init_ratio = self.args.filter_init_prune_number / filters
        per_ratio = self.args.filter_number_per_prune / filters
        self.prunable_filter_number = filters
        self.prune_record(
            f"Total prunable filters: {filters}\n"
            f"initial pruned filters: {self.args.filter_init_prune_number}({init_ratio:.2f})\n"
            f"pruned filters per interval: {self.args.filter_number_per_prune}({per_ratio:.2f})\n"
        )
        return filters

    def register_hooks(self):
        assert hasattr(self, "prunable_filters") and len(self.prunable_filters) > 0

        self.current_hooks = []
        self.filter_ranks = {}

        def forward_hook(self, input, output):
            self.output = output

        def backward_hook(self, grad_input, grad_output):
            taylor = grad_output[0] * self.output
            # Get the average value for every filter, 
            # accross all the other dimensions
            taylor = taylor.mean(dim=(0, 2, 3)).data
            self.taylor_log += taylor
            

        for name, m in self.prunable_filters.items():
            f_hook = m.register_forward_hook(forward_hook)
            b_hook = m.register_backward_hook(backward_hook)
            self.current_hooks.append(f_hook)
            self.current_hooks.append(b_hook)
            
            m.taylor_log = torch.FloatTensor(m.out_channels).zero_().cuda()


    def release_hooks(self):
        for handle in self.current_hooks:
            handle.remove()
        self.current_hooks = []

    def process_epoch(self):
        print(f"Processing one epoch...")
        self.model.eval()

        ce = CrossEntropyLabelSmooth(self.train_loader.dataset.num_classes, self.args.label_smoothing).to('cuda')
        featloss = torch.nn.MSELoss()

        for batch, label in self.train_loader:
            self.model.zero_grad()
            batch, label = batch.cuda(), label.cuda()

            loss, top1, ce_loss, feat_loss, linear_loss, l2sp_loss, total_loss = self.compute_loss(
                batch, label, 
                ce, featloss,
            )
            loss.backward()


    def normalize_ranks(self):
        self.filter_ranks = {}
        for name, m in self.prunable_filters.items():
            taylor = torch.abs(m.taylor_log)
            taylor = taylor.type(torch.FloatTensor)
            taylor = taylor / np.sqrt(torch.sum(taylor * taylor))
            self.filter_ranks[name] = taylor.cpu()
    # def normalize_ranks_per_layer(self):
    #     # print('filterprunner model',self.model)
    #     for i in self.filter_ranks:
    #         v = torch.abs(self.filter_ranks[i])
    #         v = v.type(torch.FloatTensor)
    #         v = v / np.sqrt(torch.sum(v * v))
    #         self.filter_ranks[i] = v.cpu()

    def lowest_ranking_filters(self, num):
        data = []
        for key in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[key].size(0)):
                data.append((key, j, self.filter_ranks[key][j]))
        return nsmallest(num, data, itemgetter(2))

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)
        
        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            # Because we do not do real pruning, only set channels to 0
            # So don't have to adjust the index
            # for i in range(len(filters_to_prune_per_layer[l])):
            #     filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i
        
        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        self.prune_record("Pruning plan")
        for layer in sorted(filters_to_prune_per_layer.keys()):
            self.prune_record(f"{layer}, {filters_to_prune_per_layer[layer]}")
        
        return filters_to_prune_per_layer

    def conduct_prune_resnet18(self, prunning_plan_per_layer):
        def get_nxt_bn(name):
            names = name.split('.')
            names[-1] = 'bn1'
            bn_name = ".".join(names)
            return bn_name
        
        def get_nxt_conv(name):
            names = name.split('.')
            names[-1] = 'conv2'
            conv_name = ".".join(names)
            return conv_name

        def name_to_module_exp(name):
            names = name.split('.')
            seq_exp = f"{names[0]}[{names[1]}]"
            module_exp = ".".join([seq_exp, names[-1]])
            return module_exp
        
        # Prune by set corresponding layer to 0
        for conv_name, indexes in prunning_plan_per_layer.items():
            
            nxt_bn_name = get_nxt_bn(conv_name)
            nxt_conv_name = get_nxt_conv(conv_name)

            cur_conv_exp = name_to_module_exp(conv_name)
            nxt_bn_exp = name_to_module_exp(nxt_bn_name)
            nxt_conv_exp = name_to_module_exp(nxt_conv_name)

            cur_conv = eval(f"self.model.{cur_conv_exp}")
            nxt_bn = eval(f"self.model.{nxt_bn_exp}")
            nxt_conv = eval(f"self.model.{nxt_conv_exp}")

            if len(indexes) == cur_conv.out_channels:
                raise RuntimeErroor("Conv {conv_name} is pruned all")

            cur_conv.weight.data[indexes,:,:,:] = 0
            nxt_bn.weight.data[indexes] = 0
            nxt_bn.bias.data[indexes] = 0
            nxt_bn.running_mean[indexes] = 0
            nxt_bn.running_var[indexes] = 1
            nxt_conv.weight.data[:,indexes,:,:] = 0


    def taylor_filter_prune(self, num_filters_to_prune, iteration):
        self.prune_record(
            f"Iter {iteration} prune {num_filters_to_prune} filters"
        )
        self.register_hooks()

        self.process_epoch()
        self.normalize_ranks()
        prunning_plan_per_layer = self.get_prunning_plan(num_filters_to_prune)
        self.conduct_prune_resnet18(prunning_plan_per_layer)


        self.release_hooks()


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

        total = self.prunable_filter_number
        num = total
        for name, m in model.named_modules():
            if hasattr(m, "prune_ref") and m.prune_ref:
                weight_zero = m.weight.data != 0
                out_channel = weight_zero.sum(dim=(1, 2, 3))
                pruned_channel = int((out_channel == 0).sum())
                num -= pruned_channel
                
        ratio = (total - num) / total
        log = f"===>Check filter: Total {total}, current {num}, prune ratio {ratio:2f}"
        self.prune_record(log)

    def init_prune(self):
        num = self.args.filter_init_prune_number
        self.prune_record(f"Init prune {num} filters")
        self.taylor_filter_prune(num, 0)
        self.check_param_num()

    def iterative_prune(self, iteration):
        if iteration == 0:
            return
        init = self.args.filter_init_prune_number
        interval = self.args.prune_interval
        per_num = self.args.filter_number_per_prune
        num_filter = init + per_num * (iteration / interval)
        if num_filter - self.args.filter_total_number > per_num:
            return
        
        self.logger = open(self.log_path, "a")
        num_filter = int(min(
            self.args.filter_total_number,
            num_filter
        ))
        log = f"Iteration {iteration}, prune num filter {num_filter}"
        self.prune_record(log)
        self.taylor_filter_prune(num_filter, iteration)
        self.check_param_num()
        self.logger.close()

    def final_check_param_num(self):
        self.logger = open(self.log_path, "a")
        self.check_param_num()
        self.logger.close()
        


# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=512, out_features=67, bias=True)
# )