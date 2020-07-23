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
from trigger import teacher_train

sys.path.append('../..')

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_datapath", type=str, default='../data/GTSRB', help='path to the dataset')
    parser.add_argument("--teacher_dataset", type=str, default='GTSRBData',
                        help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')
    parser.add_argument("--student_datapath", type=str, default='../data/stanford_dog', help='path to the dataset')
    parser.add_argument("--student_dataset", type=str, default='SDog120Data',
                        help='Target dataset. Currently support: \{SDog120Data, CUB200Data, Stanford40Data, MIT67Data, Flower102Data\}')

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
    parser.add_argument("--beta", type=float, default=1e-2,
                        help='The strength of the L2 regularization on the last linear layer')
    parser.add_argument("--dropout", type=float, default=0, help='Dropout rate for spatial dropout')
    parser.add_argument("--l2sp_lmda", type=float, default=0)
    parser.add_argument("--feat_lmda", type=float, default=0)
    parser.add_argument("--feat_layers", type=str, default='1234',
                        help='Used for DELTA (which layers or stages to match), ResNets should be 1234 and MobileNetV2 should be 12345')
    parser.add_argument("--reinit", action='store_true', default=False, help='Reinitialize before training')
    parser.add_argument("--no_save", action='store_true', default=False, help='Do not save checkpoints')
    parser.add_argument("--swa", action='store_true', default=False, help='Use SWA')
    parser.add_argument("--swa_freq", type=int, default=500, help='Frequency of averaging models in SWA')
    parser.add_argument("--swa_start", type=int, default=0, help='Start SWA since which iterations')
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--checkpoint", type=str, default='', help='Load a previously trained checkpoint')
    parser.add_argument("--network", type=str, default='resnet18',
                        help='Network architecture. Currently support: \{resnet18, resnet50, resnet101, mbnetv2\}')
    parser.add_argument("--shot", type=int, default=-1,
                        help='Number of training samples per class for the training dataset. -1 indicates using the '
                             'full dataset.')
    parser.add_argument("--log", action='store_true', default=False, help='Redirect the output to log/args.name.log')
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--B", type=float, default=0.1, help='Attack budget')
    parser.add_argument("--m", type=float, default=1000, help='Hyper-parameter for task-agnostic attack')
    parser.add_argument("--pgd_iter", type=int, default=40)
    parser.add_argument("--teacher_method", default=None,
                        choices=[None, "weight", "taylor_filter", "snip", "perlayer_weight",
                                 "dataset_grad", "dataset_grad_optim", "global_dataset_grad_optim",
                                 "global_dataset_grad_optim_3kiter",
                                 "global_datasetgrad_mul_mag", "global_datasetgrad_div_mag",
                                 "global_datasetgrad_div_mag_3kiter",
                                 "inv_grad_plane", "inv_grad_avg", "inv_grad_optim",
                                 "forward_backward_grad", "backdoor_finetune"]
                        )
    parser.add_argument("--student_method", default=None,
                        choices=[None, "weight", "taylor_filter", "snip", "perlayer_weight",
                                 "dataset_grad", "dataset_grad_optim", "global_dataset_grad_optim",
                                 "global_dataset_grad_optim_3kiter",
                                 "global_datasetgrad_mul_mag", "global_datasetgrad_div_mag",
                                 "global_datasetgrad_div_mag_3kiter",
                                 "inv_grad_plane", "inv_grad_avg", "inv_grad_optim",
                                 "forward_backward_grad", "backdoor_finetune"]
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
    # grad / mag
    parser.add_argument("--weight_low_bound", default=0, type=float)
    parser.add_argument("--argportion", default=0.2, type=float)
    parser.add_argument("--student_ckpt", type=str, default='')

    # Finetune for backdoor attack
    parser.add_argument("--backdoor_update_ratio", default=0, type=float,
                        help="From how much ratio does the weight update")
    parser.add_argument("--fixed_pic", default=False, action="store_true")

    args = parser.parse_args()
    if args.feat_lmda > 0:
        args.feat_lmda = -args.feat_lmda
    if args.l2sp_lmda > 0:
        args.l2sp_lmda = -args.l2sp_lmda

    args.family_output_dir = args.output_dir
    args.output_dir = osp.join(
        args.output_dir,
        args.name
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    params_out_path = osp.join(args.output_dir, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(vars(args), jf, indent=True)
    print(args)

    return args


def untarget_test(machine, test_path):
    model = machine.model
    teacher = machine.teacher
    loader = machine.test_loader
    reg_layers = machine.reg_layers
    args = machine.args
    loss = True

    with torch.no_grad():
        model.eval()

        if loss:
            teacher.eval()

            ce = CrossEntropyLabelSmooth(loader.dataset.num_classes, args.label_smoothing).to('cuda')
            featloss = torch.nn.MSELoss(reduction='none')

        total_ce = 0
        total_feat_reg = np.zeros(len(reg_layers))
        total_l2sp_reg = 0
        total = 0
        top1 = 0

        total = 0
        top1 = 0
        for i, (batch, label) in enumerate(loader):
            batch, label = batch.to('cuda'), label.to('cuda')

            total += batch.size(0)
            out = model(batch)
            _, pred = out.max(dim=1)
            top1 += int(pred.eq(label).sum().item())

            if loss:
                total_ce += ce(out, label).item()
                if teacher is not None:
                    with torch.no_grad():
                        tout = teacher(batch)

                    for key in reg_layers:
                        src_x = reg_layers[key][0].out
                        tgt_x = reg_layers[key][1].out
                        # print(src_x.shape, tgt_x.shape)

                        regloss = featloss(src_x, tgt_x.detach()).mean()

                        total_feat_reg[key] += regloss.item()

                _, unweighted = l2sp(model, 0)
                total_l2sp_reg += unweighted.item()

    test_top, clean_test_ce_loss, clean_test_feat_loss, clean_test_weight_loss, clean_test_feat_layer_loss = \
        float(top1) / total * 100, total_ce / (i + 1), np.sum(total_feat_reg) / (i + 1), total_l2sp_reg / (i + 1), \
        total_feat_reg / (i + 1)

    with open(test_path, 'a') as af:
        af.write('Start testing: untarget attack (special method)' + '\n')
        columns = ['time', 'Acc', 'celoss', 'featloss', 'l2sp']
        af.write('\t'.join(columns) + '\n')
        localtime = time.asctime(time.localtime(time.time()))[4:-6]
        test_cols = [
            localtime,
            round(test_top, 2),
            round(clean_test_ce_loss, 2),
            round(clean_test_feat_loss, 2),
            round(clean_test_weight_loss, 2),
        ]
        af.write('\t'.join([str(c) for c in test_cols]) + '\n')


def testing(data_loader, kind):
    finetune_machine.test_loader = data_loader
    test_path = osp.join(args.output_dir, "test.tsv")

    if kind != 'untarget attack':
        test_top, clean_test_ce_loss, clean_test_feat_loss, clean_test_weight_loss, clean_test_feat_layer_loss = finetune_machine.test()

        with open(test_path, 'a') as af:
            af.write('Start testing: ' + kind + '\n')
            columns = ['time', 'Acc', 'celoss', 'featloss', 'l2sp']
            af.write('\t'.join(columns) + '\n')
            localtime = time.asctime(time.localtime(time.time()))[4:-6]
            test_cols = [
                localtime,
                round(test_top, 2),
                round(clean_test_ce_loss, 2),
                round(clean_test_feat_loss, 2),
                round(clean_test_weight_loss, 2),
            ]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')
    else:
        untarget_test(finetune_machine, test_path)


def generate_dataloader(args, normalize, seed):
    # print(args.fixed_pic)
    train_set = eval(args.student_dataset)(
        args.student_datapath, True, [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=0, fixed_pic=args.fixed_pic  # !此处用原始数据集进行finetune的训练
    )

    test_set = eval(args.student_dataset)(
        args.student_datapath, False, [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=1, only_change_pic=False, fixed_pic=args.fixed_pic
    )
    #####################################################################
    test_set_1 = eval(args.student_dataset)(
        args.student_datapath, False, [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=1, only_change_pic=True, fixed_pic=args.fixed_pic
    )
    test_set_2 = eval(args.student_dataset)(
        args.student_datapath, False, [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=1, only_change_pic=False, fixed_pic=args.fixed_pic
    )
    clean_set = eval(args.student_dataset)(
        args.student_datapath, False, [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],
        args.shot, seed, preload=False, portion=0, fixed_pic=args.fixed_pic
    )
    #####################################################################
    # 测试修改图片是否成功
    # print(args.fixed_pic)
    for j in range(7):
        iii = random.randint(0, len(test_set))
        originphoto = test_set[iii][0]
        numpyphoto = np.transpose(originphoto.numpy(), (1, 2, 0))
        plt.imshow(numpyphoto)
        plt.show()
        # train_set[i][0].show()
        print(iii, test_set[iii][1])
        input()

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
    #####################################################################
    test_loader_1 = torch.utils.data.DataLoader(
        test_set_1,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )
    test_loader_2 = torch.utils.data.DataLoader(
        test_set_2,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )
    clean_loader = torch.utils.data.DataLoader(
        clean_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False
    )
    #####################################################################
    return train_loader, test_loader, test_loader_1, test_loader_2, clean_loader


if __name__ == '__main__':

    seed = 259
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = get_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Used to make sure we sample the same image for few-shot scenarios

    # 封装了生成数据的部分
    train_loader, test_loader, test_loader_1, test_loader_2, clean_loader = generate_dataloader(args, normalize, seed)

    teacher_set = eval(args.teacher_dataset)(args.teacher_datapath)

    teacher = eval('{}_dropout'.format(args.network))(
        pretrained=True,
        dropout=args.dropout,
        num_classes=teacher_set.num_classes
    ).cuda()

    if args.checkpoint == '':
        teacher_train(teacher, args)
        load_path = args.output_dir +'/teacher_ckpt.pth'
    else:
        load_path = args.checkpoint

    checkpoint = torch.load(load_path)
    teacher.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded teacher checkpoint from {args.checkpoint}")

    finetune_machine = Finetuner(
        args, teacher, teacher, test_loader_1, test_loader_1, 'chaolaichaoqu'
    )
    testing(test_loader_1, 'trigger, untarget attack')
    testing(test_loader_2, 'trigger, target attack')
    testing(clean_loader, 'clean set')
    exit(0)
    # Pre-trained model
    # 以下是错误的做法
    # student = eval('{}_dropout'.format(args.network))(
    #     pretrained=True,
    #     dropout=0,
    #     num_classes=train_loader.dataset.num_classes
    # ).cuda()

    # 更改输出的类别数目
    teacher.fc = nn.Linear(512, train_loader.dataset.num_classes)
    teacher = teacher.cuda()

    # 不能用teacher = teacher_train(teacher, args)直接传回值
    student = copy.deepcopy(teacher).cuda()

    if args.student_ckpt != '':
        checkpoint = torch.load(args.student_ckpt)
        student.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded student checkpoint from {args.checkpoint}")

    if True:
        if args.student_method == "weight":
            finetune_machine = WeightPruner(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "taylor_filter":
            finetune_machine = TaylorFilterPruner(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "snip":
            finetune_machine = SNIPPruner(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "perlayer_weight":
            finetune_machine = PerlayerWeightPruner(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "dataset_grad":
            finetune_machine = DatasetGrad(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "dataset_grad_optim":
            finetune_machine = DatasetGradOptim(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "global_dataset_grad_optim":
            finetune_machine = GlobalDatasetGradOptim(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "global_dataset_grad_optim_3kiter":
            finetune_machine = GlobalDatasetGradOptimIter(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "global_datasetgrad_mul_mag":
            finetune_machine = GlobalDatasetGradOptimMulMag(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "global_datasetgrad_div_mag":
            finetune_machine = GlobalDatasetGradOptimDivMag(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "global_datasetgrad_div_mag_3kiter":
            finetune_machine = GlobalDatasetGradOptimDivMagIter(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "inv_grad_optim":
            finetune_machine = InvGradOptim(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "inv_grad_plane":
            finetune_machine = InvGradPlane(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "inv_grad_avg":
            finetune_machine = InvGradAvg(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "forward_backward_grad":
            finetune_machine = ForwardBackwardGrad(
                args,
                student, teacher,
                train_loader, test_loader,
            )
        elif args.student_method == "backdoor_finetune":
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
                "TWO"
            )

    if args.student_ckpt == '':
        finetune_machine.train()

    if args.student_method is not None:
        finetune_machine.final_check_param_num()

    testing(test_loader_1, 'trigger, untarget attack')
    testing(test_loader_2, 'trigger, target attack')
    testing(clean_loader, 'clean set')
