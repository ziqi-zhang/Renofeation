import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchcontrib

from torchvision import transforms

from dataset.cub200 import CUB200Data
from dataset.mit67 import MIT67Data
from dataset.stanford_dog import SDog120Data
from dataset.caltech256 import Caltech257Data
from dataset.stanford_40 import Stanford40Data
from dataset.flower102 import Flower102Data
from dataset.gtsrb import GTSRBData

sys.path.append('..')
from model.fe_resnet import resnet18_dropout, resnet50_dropout, resnet101_dropout
from model.fe_mobilenet import mbnetv2_dropout
from model.fe_resnet import feresnet18, feresnet50, feresnet101
from model.fe_mobilenet import fembnetv2

from eval_robustness import advtest, myloss
from utils import *

from fineprune.weight_pruner import WeightPruner
from fineprune.perlayer_weight_pruner import PerlayerWeightPruner
from fineprune.taylor_filter_pruner import TaylorFilterPruner
from fineprune.snip import SNIPPruner
from fineprune.dataset_grad import DatasetGrad
# from fineprune.dataset_grad_optim import DatasetGradOptim
# from fineprune.global_datasetgrad_optim import GlobalDatasetGradOptim
# from fineprune.global_datasetgrad_optim_iter import GlobalDatasetGradOptimIter
# from fineprune.global_dataset_grad_mul_mag import GlobalDatasetGradOptimMulMag
# from fineprune.global_dataset_grad_div_mag import GlobalDatasetGradOptimDivMag
# from fineprune.global_dataset_grad_div_mag_iter import GlobalDatasetGradOptimDivMagIter
from fineprune.inv_grad_optim import InvGradOptim
from fineprune.inv_grad import *
from fineprune.forward_backward_grad import ForwardBackwardGrad

from backdoor.attack_finetuner import AttackFinetuner
from backdoor.prune import weight_prune
from backdoor.finetuner import Finetuner


def teacher_train(teacher, args):
    seed = 98
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Used to make sure we sample the same image for few-shot scenarios
    seed = 98

    train_set = eval(args.teacher_dataset)(
        args.teacher_datapath, True, [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=args.argportion, fixed_pic=args.fixed_pic
    )
    test_set = eval(args.teacher_dataset)(
        args.teacher_datapath, False, [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],  # target attack
        args.shot, seed, preload=False, portion=1, only_change_pic=False, fixed_pic=args.fixed_pic, four_corner=args.four_corner
    )
    clean_set = eval(args.teacher_dataset)(
        args.teacher_datapath, False, [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=0, fixed_pic=args.fixed_pic
    )

    # print("trigger py file",args.argportion)
    # for j in range(20):
    #     iii = random.randint(0, len(train_set))
    #     originphoto = train_set[iii][0]
    #     # originphoto = originphoto.numpy() * normalize.std + normalize.mean
    #     numpyphoto = np.transpose(originphoto.numpy(), (1, 2, 0))
    #     # numpyphoto = numpyphoto * normalize.std + normalize.mean
    #     plt.imshow(numpyphoto)
    #     plt.show()
    #     print(iii, train_set[iii][1],"teacher")
    #     input()

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
    clean_loader = torch.utils.data.DataLoader(
        clean_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )

    # input()
    student = copy.deepcopy(teacher).cuda()
    if True:
        if args.teacher_method == "weight":
            finetune_machine = WeightPruner(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "taylor_filter":
            finetune_machine = TaylorFilterPruner(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "snip":
            finetune_machine = SNIPPruner(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "perlayer_weight":
            finetune_machine = PerlayerWeightPruner(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "dataset_grad":
            finetune_machine = DatasetGrad(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "dataset_grad_optim":
            finetune_machine = DatasetGradOptim(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "global_dataset_grad_optim":
            finetune_machine = GlobalDatasetGradOptim(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "global_dataset_grad_optim_3kiter":
            finetune_machine = GlobalDatasetGradOptimIter(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "global_datasetgrad_mul_mag":
            finetune_machine = GlobalDatasetGradOptimMulMag(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "global_datasetgrad_div_mag":
            finetune_machine = GlobalDatasetGradOptimDivMag(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "global_datasetgrad_div_mag_3kiter":
            finetune_machine = GlobalDatasetGradOptimDivMagIter(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "inv_grad_optim":
            finetune_machine = InvGradOptim(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "inv_grad_plane":
            finetune_machine = InvGradPlane(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "inv_grad_avg":
            finetune_machine = InvGradAvg(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "forward_backward_grad":
            finetune_machine = ForwardBackwardGrad(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.teacher_method == "backdoor_finetune":
            student = weight_prune(
                student, args.backdoor_update_ratio,
            )
            finetune_machine = AttackFinetuner(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        else:
            finetune_machine = Finetuner(
                args,
                student, teacher,
                train_loader, test_loader,
                "ONE"
            )

    finetune_machine.train()
    # finetune_machine.adv_eval()

    # if args.teacher_method is not None:
    #    finetune_machine.final_check_param_num()

    # start testing (more testing, more cases)
    finetune_machine.test_loader = test_loader

    test_top1, test_ce_loss, test_feat_loss, test_weight_loss, test_feat_layer_loss = finetune_machine.test()
    test_path = osp.join(args.output_dir, "test.tsv")

    with open(test_path, 'a') as af:
        af.write('Teacher! Start testing:    trigger dataset:\n')
        columns = ['time', 'Acc', 'celoss', 'featloss', 'l2sp']
        af.write('\t'.join(columns) + '\n')
        localtime = time.asctime(time.localtime(time.time()))[4:-6]
        test_cols = [
            localtime,
            round(test_top1, 2),
            round(test_ce_loss, 2),
            round(test_feat_loss, 2),
            round(test_weight_loss, 2),
        ]
        af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    finetune_machine.test_loader = clean_loader
    test_top2, clean_test_ce_loss, clean_test_feat_loss, clean_test_weight_loss, clean_test_feat_layer_loss = finetune_machine.test()
    test_path = osp.join(args.output_dir, "test.tsv")

    with open(test_path, 'a') as af:
        af.write('Teacher! Start testing:    clean dataset:\n')
        columns = ['time', 'Acc', 'celoss', 'featloss', 'l2sp']
        af.write('\t'.join(columns) + '\n')
        localtime = time.asctime(time.localtime(time.time()))[4:-6]
        test_cols = [
            localtime,
            round(test_top2, 2),
            round(clean_test_ce_loss, 2),
            round(clean_test_feat_loss, 2),
            round(clean_test_weight_loss, 2),
        ]
        af.write('\t'.join([str(c) for c in test_cols]) + '\n')
    return student
