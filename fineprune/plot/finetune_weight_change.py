import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random
import seaborn as sns


import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcontrib

from torchvision import transforms

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
from fineprune.weight_pruner import WeightPruner
from fineprune.perlayer_weight_pruner import PerlayerWeightPruner
from fineprune.taylor_filter_pruner import TaylorFilterPruner
from fineprune.snip import SNIPPruner
from fineprune.dataset_grad import DatasetGrad
from fineprune.local_datasetgrad_optim_epoch import LocalDatasetGradOptimEpoch
from fineprune.global_datasetgrad_optim_epoch import GlobalDatasetGradOptimEpoch
from fineprune.global_datasetgrad_optim_iter import GlobalDatasetGradOptimIter
from fineprune.global_datasetgrad_optim_iter_postweight import GlobalDatasetGradOptimDivMagIterPostweight
from fineprune.global_datasetgrad_mulmag import GlobalDatasetGradOptimMulMag
from fineprune.global_datasetgrad_divmag_epoch import GlobalDatasetGradOptimDivMagEpoch
from fineprune.global_datasetgrad_divmag_iter import GlobalDatasetGradOptimDivMagIter
from fineprune.inv_grad_optim import InvGradOptim
from fineprune.inv_grad import *
from fineprune.forward_backward_grad import ForwardBackwardGrad
from fineprune.divmag_avg import GlobalDatasetGradOptimDivMagIterAvg

from matplotlib import pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='/data', help='path to the dataset')
    parser.add_argument("--dataset", type=str, default='CUB200Data', help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')
    parser.add_argument("--iterations", type=int, default=30000, help='Iterations to train')
    parser.add_argument("--print_freq", type=int, default=100, help='Frequency of printing training logs')
    parser.add_argument("--test_interval", type=int, default=1000, help='Frequency of testing')
    parser.add_argument("--adv_test_interval", type=int, default=1000)
    parser.add_argument("--name", type=str, default='test', help='Name for the checkpoint')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--const_lr", action='store_true', default=False, help='Use constant learning rate')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=1e-2, help='The strength of the L2 regularization on the last linear layer')
    parser.add_argument("--dropout", type=float, default=0, help='Dropout rate for spatial dropout')
    parser.add_argument("--l2sp_lmda", type=float, default=0)
    parser.add_argument("--feat_lmda", type=float, default=0)
    parser.add_argument("--feat_layers", type=str, default='1234', help='Used for DELTA (which layers or stages to match), ResNets should be 1234 and MobileNetV2 should be 12345')
    parser.add_argument("--reinit", action='store_true', default=False, help='Reinitialize before training')
    parser.add_argument("--no_save", action='store_true', default=False, help='Do not save checkpoints')
    parser.add_argument("--swa", action='store_true', default=False, help='Use SWA')
    parser.add_argument("--swa_freq", type=int, default=500, help='Frequency of averaging models in SWA')
    parser.add_argument("--swa_start", type=int, default=0, help='Start SWA since which iterations')
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--checkpoint", type=str, default='', help='Load a previously trained checkpoint')
    parser.add_argument("--network", type=str, default='resnet18', help='Network architecture. Currently support: \{resnet18, resnet50, resnet101, mbnetv2\}')
    parser.add_argument("--shot", type=int, default=-1, help='Number of training samples per class for the training dataset. -1 indicates using the full dataset.')
    parser.add_argument("--log", action='store_true', default=False, help='Redirect the output to log/args.name.log')
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--B", type=float, default=0.1, help='Attack budget')
    parser.add_argument("--m", type=float, default=1000, help='Hyper-parameter for task-agnostic attack')
    parser.add_argument("--pgd_iter", type=int, default=40)
    parser.add_argument("--method", default=None, 
        # choices=[None, "weight", "taylor_filter", "snip", "perlayer_weight",
        # "dataset_grad", "dataset_grad_optim", "global_dataset_grad_optim", "global_dataset_grad_optim_iter",
        # "global_datasetgrad_mul_mag", "global_datasetgrad_div_mag", "global_datasetgrad_div_mag_iter",
        # "inv_grad_plane", "inv_grad_avg", "inv_grad_optim",
        # "forward_backward_grad"]
    )
    parser.add_argument("--train_all", default=False, action="store_true")
    parser.add_argument("--lrx10", default=True)
    parser.add_argument("--prune_interval", default=-1, type=int)
    
    # weight dist
    parser.add_argument("--finetune_ckpt", type=str, default='')
    parser.add_argument("--retrain_ckpt", type=str, default='')
    parser.add_argument("--renofeation_ckpt", type=str, default='')
    parser.add_argument("--my_ckpt", type=str, default='')
    args = parser.parse_args()
    if args.feat_lmda > 0:
        args.feat_lmda = -args.feat_lmda
    if args.l2sp_lmda > 0:
        args.l2sp_lmda = -args.l2sp_lmda

    args.family_output_dir = args.output_dir
    args.output_dir = osp.join(
        args.output_dir, args.dataset
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    params_out_path = osp.join(args.output_dir, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(vars(args), jf, indent=True)
    print(args)

    return args

def plot_weights(finetune, retrain, reno, my, args):
    
    bins = 100
    y, edges, _ = plt.hist(finetune, bins=bins, alpha=0.3, color="blue",
                        range=(0, 2))
    bincenters = 0.5*(edges[1:]+edges[:-1])
    plt.plot(bincenters,y, linestyle='solid', alpha=1, color="blue", label='Fine-tune')
    # sns.distplot(finetune, bins=bins, label='Fine-tune')

    y, edges, _ = plt.hist(retrain, bins=bins, alpha=0.3, color="red",
            range=(0, 2))
    bincenters = 0.5*(edges[1:]+edges[:-1])
    plt.plot(bincenters,y, linestyle='dotted', alpha=1, color="red", label='Retrain')
    # sns.distplot(retrain, bins=bins, label='Retrain')
    # plt.hist(reno, bins=bins, alpha=0.3, range=(0, 2), label='Renofeation')
    y, edges, _ = plt.hist(my, bins=bins, alpha=0.3, color="green",
            range=(0, 2))
    bincenters = 0.5*(edges[1:]+edges[:-1])
    plt.plot(bincenters,y, linestyle='dashed', alpha=1, color="green", label='SelF')
    # sns.distplot(my, bins=bins, label='Our approach')
    # plt.title('Weight difference')
    plt.xlabel(r'Weight change $|\frac{w^{S}}{w^{T}}|$', fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper right', prop={'size': 15})
    # plt.xlim(-0.5, 0.5)
    plt.ylim(0, 1e6)
    plt.tight_layout()

    path = osp.join(args.output_dir, "weight.pdf")
    plt.savefig(path)

def load_student(ckpt, args, num_classes):
    model = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=args.dropout, 
        num_classes=num_classes
    ).cuda()
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint from {ckpt}")
    model.eval()
    return model

def compute_weight_diff(teacher, student, args):
    conv_dict = {}
    weight_diff = []
    for name, module in teacher.named_modules():
        if ( isinstance(module, nn.Conv2d) ):
            conv_dict[name] = [module]
    for name, module in student.named_modules():
        if ( isinstance(module, nn.Conv2d) ):
            conv_dict[name].append(module)

    for name, modules in conv_dict.items():
        m_teacher, m_student = modules
        change = (m_student.weight / m_teacher.weight).abs().detach().cpu().flatten().numpy().tolist()
        weight_diff += change
    
    return weight_diff

if __name__=="__main__":
    seed = 98
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = get_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Used to make sure we sample the same image for few-shot scenarios
    seed = 98

    train_set = eval(args.dataset)(
        args.datapath, True, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), 
        args.shot, seed, preload=False
    )
    test_set = eval(args.dataset)(
        args.datapath, False, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), 
        args.shot, seed, preload=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )

    retrain = load_student(args.retrain_ckpt, args, train_loader.dataset.num_classes)
    finetune = load_student(args.finetune_ckpt, args, train_loader.dataset.num_classes)
    renofeation = load_student(args.renofeation_ckpt, args, train_loader.dataset.num_classes)
    my_model = load_student(args.my_ckpt, args, train_loader.dataset.num_classes)

    # Pre-trained model
    teacher = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=0, 
        num_classes=train_loader.dataset.num_classes
    ).cuda()

    finetune_diff = compute_weight_diff(teacher, finetune, args)
    retrain_diff = compute_weight_diff(teacher, retrain, args)
    renofeation_diff = compute_weight_diff(teacher, renofeation, args)
    my_diff = compute_weight_diff(teacher, my_model, args)

    plot_weights(finetune_diff, retrain_diff, renofeation_diff, my_diff, args)

