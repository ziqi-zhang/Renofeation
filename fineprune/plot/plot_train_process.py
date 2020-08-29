import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random
import pandas as pd

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
    parser.add_argument("--finetune_dir", type=str, default='')
    parser.add_argument("--retrain_dir", type=str, default='')
    parser.add_argument("--renofeation1_dir", type=str, default='')
    parser.add_argument("--renofeation2_dir", type=str, default='')
    parser.add_argument("--my_dir", type=str, default='')
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

def plot_process(finetune, retrain, reno, my):

    plt.plot(finetune.iter.tolist(), finetune.Acc.tolist(), label="Fine-tune",
            linewidth=2, linestyle="dashdot", color="green")
    plt.plot(retrain.iter.tolist(), retrain.Acc.tolist(), label="Retrain",
            linewidth=2, linestyle="dotted", color="red")
    plt.plot(reno.iter.tolist(), reno.Acc.tolist(), label="Renofeation",
            linewidth=2, linestyle="dashed", color="gray")
    plt.plot(my.iter.tolist(), my.Acc.tolist(), label="Our approach",
            linewidth=2, linestyle="solid", color="blue")
    
    plt.plot([0, 3000], [0, 0], color='black', linewidth=3, marker='*')
    plt.annotate('Trial-tune', xy=(3300, 2), xytext=(10000, 10),
            xycoords='data',
            arrowprops=dict(facecolor='black', shrink=1),
            fontsize=25,
            )


    # plt.title('Train process')
    plt.xlabel('Iteration', fontsize=25)
    plt.xticks(fontsize=25)
    plt.xticks((20000, 40000, 60000, 80000), ("20K", "40K", "60K", "80K"))
    plt.ylabel('Accuracy (ACC)', fontsize=25)
    plt.yticks(fontsize=25)
    # plt.legend(loc='lower right', prop={'size': 20})
    # plt.xlim(-0.5, 0.5)
    # plt.ylim(0, 2e5)
    # plt.tight_layout()

    # path = osp.join(args.output_dir, f"{args.dataset}_acc.pdf")
    # plt.savefig(path)
    # plt.clf()

def plot_process_adv(finetune, retrain, reno, my):

    plt.plot(finetune.iter.tolist(), finetune.ASR.tolist(), label="Fine-tune",
            linewidth=2, linestyle="dashdot", color="green")
    plt.plot(retrain.iter.tolist(), retrain.ASR.tolist(), label="Retrain",
            linewidth=2, linestyle="dotted", color="red")
    plt.plot(reno.iter.tolist(), reno.ASR.tolist(), label="Renofeation",
            linewidth=2, linestyle="dashed", color="gray")
    plt.plot(my.iter.tolist(), my.ASR.tolist(), label="SelF",
            linewidth=2, linestyle="solid", color="blue")
    
    plt.plot([0, 3000], [10, 10], color='black', linewidth=3, marker='*')
    plt.annotate('Trial-tune', xy=(0, 12), xytext=(0, 20),
            xycoords='data',
            arrowprops=dict(facecolor='black', shrink=1),
            fontsize=25,
            )

    # plt.title('Train process')
    plt.xlabel('Iteration', fontsize=25)
    plt.xticks(fontsize=25)
    plt.xticks((20000, 40000, 60000, 80000), ("20K", "40K", "60K", "80K"))
    plt.ylabel('Defect inheritance rate (DIR)', fontsize=18)
    plt.yticks(fontsize=25)
    plt.legend(loc='upper right', prop={'size': 20})
    # plt.xlim(-0.5, 0.5)
    # plt.ylim(0, 2e5)
    # plt.tight_layout()

    # path = osp.join(args.output_dir, f"{args.dataset}_dir.pdf")
    # plt.savefig(path)
    # plt.clf()

def load_test_tsv(dir):
    path = osp.join(dir, "test.tsv")
    assert osp.exists(path)
    log = pd.read_csv(path, sep='\t', header=0)
    return log

def load_adv_test_tsv(dir):
    path = osp.join(dir, "adv.tsv")
    assert osp.exists(path)
    log = pd.read_csv(path, sep='\t', header=0)
    return log

if __name__=="__main__":
    seed = 98
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = get_args()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)

    finetune_tsv = load_test_tsv(args.finetune_dir)
    retrain_tsv = load_test_tsv(args.retrain_dir)
    reno1_tsv = load_test_tsv(args.renofeation1_dir)
    # reno2_tsv = load_test_tsv(args.renofeation2_dir)
    # reno2_tsv.iter += 90000
    # reno_tsv = pd.concat([reno1_tsv, reno2_tsv])
    my_tsv = load_test_tsv(args.my_dir)
    my_tsv.iter += 3000
    # my_tsv = my_tsv[:-3]

    plot_process(finetune_tsv, retrain_tsv, reno1_tsv, my_tsv)


    plt.subplot(122)
    
    finetune_tsv = load_adv_test_tsv(args.finetune_dir)
    retrain_tsv = load_adv_test_tsv(args.retrain_dir)
    reno1_tsv = load_adv_test_tsv(args.renofeation1_dir)
    # reno2_tsv = load_adv_test_tsv(args.renofeation2_dir)
    # reno2_tsv.iter += 90000
    # reno_tsv = pd.concat([reno1_tsv, reno2_tsv])
    my_tsv = load_adv_test_tsv(args.my_dir)
    my_tsv.iter += 3000
    my_tsv = my_tsv[:-1]

    plot_process_adv(finetune_tsv, retrain_tsv, reno1_tsv, my_tsv)
    
    plt.tight_layout()
    path = osp.join(args.output_dir, f"{args.dataset}_acc_dir.pdf")
    plt.savefig(path)
