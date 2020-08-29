import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random
import copy

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
from fineprune.mid_weight_pruner import MidWeightPruner
from fineprune.mid_datasetgrad_optim import MidDeltaW

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
    # Weight prune
    parser.add_argument("--weight_total_ratio", default=-1, type=float)
    parser.add_argument("--weight_ratio_per_prune", default=-1, type=float)
    parser.add_argument("--weight_init_prune_ratio", default=-1, type=float)
    # Taylor filter prune
    parser.add_argument("--filter_total_number", default=-1, type=int)
    parser.add_argument("--filter_number_per_prune", default=-1, type=int)
    parser.add_argument("--filter_init_prune_number", default=-1, type=int)
    # Trial finetune
    parser.add_argument("--trial_iteration", default=1000, type=int)
    parser.add_argument("--trial_lr", default=1e-2, type=float)
    parser.add_argument("--trial_momentum", default=0.9, type=float)
    parser.add_argument("--trial_weight_decay", default=0, type=float)
    # grad / mag
    parser.add_argument("--weight_low_bound", default=0, type=float)
    parser.add_argument("--prune_descending", default=False, action="store_true")

    parser.add_argument("--adjust_iteration", default=0, type=int)

    args = parser.parse_args()
    if args.feat_lmda > 0:
        args.feat_lmda = -args.feat_lmda
    if args.l2sp_lmda > 0:
        args.l2sp_lmda = -args.l2sp_lmda

    args.family_output_dir = args.output_dir
    # args.output_dir = osp.join(
    #     args.output_dir, args.dataset
    # )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    params_out_path = osp.join(args.output_dir, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(vars(args), jf, indent=True)
    print(args)

    return args

def plot_weights(retrain, finetune, renofeation, my_model, args):

    indices = list(range(len(retrain)))
    retrain_list = [v for v in retrain.values()]
    finetune_list = [v for v in finetune.values()]
    renofeation_list = [v for v in renofeation.values()]
    my_list = [v for v in my_model.values()]

    plt.plot(indices, retrain_list, label="Retrain")
    plt.plot(indices, finetune_list, label="Finetune")
    plt.plot(indices, my_list, label="SFTF")
    plt.plot(indices, renofeation_list, label="Renofeation")

    
    plt.title('Weight distance')
    plt.xlabel('Layer depth')
    plt.ylabel('Absolute distance')
    plt.legend(loc='upper right')
    # plt.xlim(-0.5, 0.5)
    # plt.ylim(0, 2e5)

    path = osp.join(args.output_dir, f"{args.dataset}.pdf")
    plt.savefig(path)
    
def load_student(ckpt, args, num_classes, test_loader):
    model = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=args.dropout, 
        num_classes=num_classes
    ).cuda()
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint from {ckpt}")
    model.eval()
    
    # total, raw_correct = 0, 0
    # for batch, label in test_loader:
    #     batch, label = batch.cuda(), label.cuda()
    #     out = model(batch)
    #     _, raw_pred = out.max(dim=1)

    #     total += int(batch.shape[0])
    #     raw_correct += int((label == raw_pred).sum())
    # correct_ratio = raw_correct / total
    # print(f"Student correct ratio {correct_ratio:.2f}")
    
    return model

def eval_model(model, ref_model, train_loader, test_loader, args, adversary=None):
    raw_model = copy.deepcopy(model)

    conv_dict = {}
    for name, module in ref_model.named_modules():
        if ( isinstance(module, nn.Conv2d) ):
            conv_dict[name] = module
    with torch.no_grad():
        for name, module in model.named_modules():
            if ( isinstance(module, nn.Conv2d) ):
                mask = conv_dict[name].weight == 0
                module.weight[mask] = 0

    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        # momentum=args.momentum, 
        # weight_decay=args.weight_decay,
    )
    ce = CrossEntropyLabelSmooth(train_loader.dataset.num_classes, args.label_smoothing).to('cuda')

    # model.train()
    dataloader_iterator = iter(train_loader)
    for i in range(args.adjust_iteration):
        try:
            batch, label = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(train_loader)
            batch, label = next(dataloader_iterator)
        
        optimizer.zero_grad()
        batch = batch.cuda()
        with torch.no_grad():
            out = raw_model(batch)
            _, raw_pred = out.max(dim=1)
        
        out = model(batch)
        loss = ce(out, raw_pred)
        loss.backward()
        optimizer.step()
        
    model.eval()
    raw_model.eval()
    total, raw_correct, del_correct = 0, 0, 0
    for batch, label in test_loader:
        batch, label = batch.cuda(), label.cuda()
        out = raw_model(batch)
        _, raw_pred = out.max(dim=1)
        out = model(batch)
        _, del_pred = out.max(dim=1)

        total += int(batch.shape[0])
        del_correct += int((label == del_pred).sum())
        raw_correct += int((label == raw_pred).sum())
        
    raw_correct_ratio = raw_correct / total
    del_correct_ratio = del_correct / total

    if adversary is not None:
        total, raw_adv_correct, del_adv_correct = 0, 0, 0
        raw_dir_correct, del_dir_correct = 0, 0
        for batch, label in test_loader:
            batch = batch.cuda()
            label = label.cuda()
            out = raw_model(batch)
            _, raw_pred_normal = out.max(dim=1)
            out = model(batch)
            _, del_pred_normal = out.max(dim=1)

            if 'mbnetv2' in args.network:
                y = torch.zeros(batch.shape[0], model.classifier[1].in_features).cuda()
            else:
                y = torch.zeros(batch.shape[0], model.fc.in_features).cuda()
            
            y[:,0] = args.m
            advbatch = adversary.perturb(batch, y)

            raw_out_adv = raw_model(advbatch)
            _, raw_pred_adv = raw_out_adv.max(dim=1)
            del_out_adv = model(advbatch)
            _, del_pred_adv = del_out_adv.max(dim=1)

            total += int(batch.shape[0])
            raw_adv_correct += int((raw_pred_normal == raw_pred_adv).sum())
            raw_dir_correct += int( ((raw_pred_normal == label) * (raw_pred_normal != raw_pred_adv)).sum() )
            del_adv_correct += int((del_pred_normal == del_pred_adv).sum())
            del_dir_correct += int( ((del_pred_normal == label) * (del_pred_normal != del_pred_adv)).sum() )
            
        raw_adv_correct_ratio = raw_adv_correct / total
        del_adv_correct_ratio = del_adv_correct / total
        raw_dir_ratio = raw_dir_correct / raw_correct
        del_dir_ratio = del_dir_correct / del_correct
        return raw_adv_correct_ratio, del_adv_correct_ratio, raw_dir_ratio, del_dir_ratio
    else:
        return raw_correct_ratio, del_correct_ratio



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

    teacher = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=0, 
        num_classes=train_loader.dataset.num_classes
    ).cuda()
    model = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=args.dropout, 
        num_classes=train_loader.dataset.num_classes
    ).cuda()
    
    if args.method == "weight":
        finetune_machine = WeightPruner(
            args,
            model, teacher,
            train_loader, test_loader,
        )
    elif args.method == "mid_weight":
        finetune_machine = MidWeightPruner(
            args,
            model, teacher,
            train_loader, test_loader,
        )
    elif args.method == "mid_deltaw":
        finetune_machine = MidDeltaW(
            args,
            model, teacher,
            train_loader, test_loader,
        )
    elif args.method == "global_dataset_grad_optim_iter":
        finetune_machine = GlobalDatasetGradOptimIter(
            args,
            model, teacher,
            train_loader, test_loader,
        )
    elif args.method == "global_datasetgrad_divmag_iter":
        finetune_machine = GlobalDatasetGradOptimDivMagIter(
            args,
            model, teacher,
            train_loader, test_loader,
        )

    test_teacher = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=0, 
    ).cuda()
    test_teacher.eval()
    test_student = load_student(args.checkpoint, args, train_loader.dataset.num_classes, test_loader)
    test_student.eval()
    ref_model = finetune_machine.model

    pretrained_model = eval('fe{}'.format(args.network))(pretrained=True).cuda().eval()
    adversary = LinfPGDAttack(
        pretrained_model, loss_fn=myloss, eps=args.B,
        nb_iter=args.pgd_iter, eps_iter=0.01, 
        rand_init=True, clip_min=-2.2, clip_max=2.2,
        targeted=False
    )

    save_path = osp.join(args.output_dir, f"{args.method}_{args.weight_total_ratio}.log")
    f = open(save_path, "w")
    raw_correct, del_correct = eval_model(test_student, ref_model, train_loader, test_loader, args)
    log = f"Student raw correct ratio {raw_correct:.2f}, del correct ratio {del_correct:.2f}"
    print(log)
    f.write(log+"\n")
    
    test_student = load_student(args.checkpoint, args, train_loader.dataset.num_classes, test_loader)
    test_student.eval()
    raw_adv_correct, del_adv_correct, raw_dir, del_dir = eval_model(test_student, ref_model, train_loader, test_loader, args, adversary)
    log = (
        f"raw adv correct ratio {raw_adv_correct:.2f}, "
        f"del adv correct ratio {del_adv_correct:.2f}.\n "
        f"raw dir ratio {raw_dir:.2f}, "
        f"del dir ratio {del_dir:.2f}."
    )
    print(log)
    f.write(log+"\n")
    f.close()

    # retrain_diff = compute_weight_diff(teacher, retrain, args)
    # finetune_diff = compute_weight_diff(teacher, finetune, args)
    # renofeation_diff = compute_weight_diff(teacher, renofeation, args)
    # my_diff = compute_weight_diff(teacher, my_model, args)
    
    # plot_weights(retrain_diff, finetune_diff,renofeation_diff, my_diff, args)

