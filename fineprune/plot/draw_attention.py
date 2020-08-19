import os
import os.path as osp
import sys
import time
import argparse
from pdb import set_trace as st
import json
import random
import matplotlib.cm as mpl_color_map
from PIL import Image


import torch
import numpy as np
import torchvision
from torchvision import transforms
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

class UnNormalize(object):
    # 去除normalization，
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def apply_colormap_on_image(org_im, activation, colormap_name='bwr'):
    # 直接用就行
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.6
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return heatmap_on_image

def load_student(ckpt, args, num_classes):
    # 读取student模型
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

# Used to matching features
def forward_hook(self, input, output):
    self.out = output

def register_hooks(model, layer_names):
    current_hooks = {}
    for name, module in model.named_modules():
        if name in layer_names:
            f_hook = module.register_forward_hook(forward_hook)
            current_hooks[name] = f_hook

    return current_hooks

def extract_cam(model, batch, attention_layer_names):
    # 把模型对batch预测时中间的feature map提取出来
    img = batch[0].clone()
    # 保存原始图片
    pil_img = unnormalize(img)
    pil_img = transforms.ToPILImage()(pil_img.cpu().detach()).convert("RGB")
    cams = []
    for name, module in model.named_modules():
        # 只提取目标layer的输出
        if name in attention_layer_names:
            # 获取feature map并取mean
            cam = module.out[0].abs().mean(0)
            cam = cam.unsqueeze(0).unsqueeze(0)
            cam = cam.squeeze().cpu().detach().numpy()
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)
            # resize
            cam = np.uint8(
                Image.fromarray(cam).resize(
                    (img.shape[1], img.shape[2]), 
                    Image.ANTIALIAS
                )
            )/255
            # 把feature map与原始图片画到一起
            cam_img = apply_colormap_on_image(pil_img, cam)
            cams.append(cam_img)
    all_cams = np.hstack( (np.asarray(img) for img in cams ) )
    all_cams = Image.fromarray(all_cams)
    return all_cams

def draw_cmp_cam(model, batch, advbatch, attention_layer_names):
    # 正常样本输入模型，然后提取attention map
    out = model(batch)
    normal_cam = extract_cam(model, batch, attention_layer_names)
    # 对抗样本输入模型，然后提取attention map
    # 你要做的是把这里的对抗样本换成带有trigger的样本
    out = model(advbatch)
    adv_cam = extract_cam(model, batch, attention_layer_names)
    # 把两个attention map组合
    cams = [normal_cam, adv_cam]
    all_cams = np.hstack( (np.asarray(img) for img in cams ) )
    all_cams = Image.fromarray(all_cams)
    return all_cams

def draw_attention(
    dataloader, adversary, args, unnormalize,
    teacher, finetune, retrain, renofeation, my_model
):
    # attention_layer_names = []
    # for name, module in teacher.named_modules():
    #     if ( isinstance(module, nn.Sequential) and "downsample" not in name):
    #         attention_layer_names.append(name)
    # 这里只提取网络在layer4的feature map作为attention
    attention_layer_names = ["layer4"]
    models = [teacher, finetune, retrain, renofeation, my_model]
    # 对所有模型register_hook，这样才能获取网络中间的feature map
    model_hooks = []
    for model in models:
        hooks = register_hooks(model, attention_layer_names)
        model_hooks.append(hooks)
        

    attention_dict = {}
    for name, module in teacher.named_modules():
        if name in attention_layer_names:
            attention_dict[name] = []
    
    for name, module in teacher.named_modules():
        if name in attention_layer_names:
            attention_dict[name].append(module)

    for i, (batch, label) in enumerate(dataloader):
        batch, label = batch.cuda(), label.cuda()
        assert batch.shape[0] == 1

        if 'mbnetv2' in args.network:
            y = torch.zeros(batch.shape[0], teacher.classifier[1].in_features).cuda()
        else:
            y = torch.zeros(batch.shape[0], teacher.fc.in_features).cuda()
        y[:,0] = args.m
        advbatch = adversary.perturb(batch, y)
        # 这里是筛选一下数据，确保网络能对正常样本做出正确判断，但是对对抗样本做出错误判断
        out = finetune(batch)
        _, pred = out.max(dim=1)
        out = finetune(advbatch)
        _, advpred = out.max(dim=1)
        out = retrain(batch)
        _, retrain_pred = out.max(dim=1)
        if (pred == advpred) or retrain_pred != label:
            continue
        # 画出各个模型的attention map
        teacher_cam = draw_cmp_cam(teacher, batch, advbatch, attention_layer_names)
        finetune_cam = draw_cmp_cam(finetune, batch, advbatch, attention_layer_names)
        retrain_cam = draw_cmp_cam(retrain, batch, advbatch, attention_layer_names)
        renofeation_cam = draw_cmp_cam(renofeation, batch, advbatch, attention_layer_names)
        my_cam = draw_cmp_cam(my_model, batch, advbatch, attention_layer_names)
        cams = [
            teacher_cam, finetune_cam, retrain_cam, renofeation_cam, my_cam
        ]

        # out = teacher(batch)
        # normal_cam = extract_cam(teacher, batch, attention_layer_names)
        # out = teacher(advbatch)
        # adv_cam = extract_cam(teacher, batch, attention_layer_names)
        # out = finetune(advbatch)
        # finetune_cam = extract_cam(finetune, batch, attention_layer_names)
        # out = retrain(advbatch)
        # retrain_cam = extract_cam(retrain, batch, attention_layer_names)
        # out = renofeation(advbatch)
        # renofeation_cam = extract_cam(renofeation, batch, attention_layer_names)
        # out = my_model(advbatch)
        # my_cam = extract_cam(my_model, batch, attention_layer_names)
        # cams = [
        #     normal_cam, adv_cam, finetune_cam, retrain_cam, renofeation_cam, my_cam
        # ]
        # 把不同模型的attention map组合并保存
        label = int(label)
        if hasattr(dataloader.dataset, "cls_names"):
            name = dataloader.dataset.cls_names[label]
        else:
            name = str(label)
        all_cams = np.vstack( (np.asarray(img) for img in cams ) )
        all_cams = Image.fromarray(all_cams)
        save_path = osp.join(args.output_dir, f"{i}_{name}.png")
        all_cams.save(save_path)
        # 这里为了方便只看前40张图片
        if i > 300:
            break

        # break
    for hooks in model_hooks:
        for handle in hooks.values():
            handle.remove()

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
    unnormalize = UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) 
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

    # Pre-trained model
    teacher = eval('{}_dropout'.format(args.network))(
        pretrained=True, 
        dropout=0, 
        num_classes=train_loader.dataset.num_classes
    ).cuda()
    # 读取生成对抗样本的对象
    pretrained_model = eval('fe{}'.format(args.network))(pretrained=True).cuda().eval()
    adversary = LinfPGDAttack(
        pretrained_model, loss_fn=myloss, eps=args.B,
        nb_iter=args.pgd_iter, eps_iter=0.01, 
        rand_init=True, clip_min=-2.2, clip_max=2.2,
        targeted=False
    )
    # 读取四个不同的模型，这里的模型是在其他地方先训好
    retrain = load_student(args.retrain_ckpt, args, train_loader.dataset.num_classes)
    finetune = load_student(args.finetune_ckpt, args, train_loader.dataset.num_classes)
    renofeation = load_student(args.renofeation_ckpt, args, train_loader.dataset.num_classes)
    my_model = load_student(args.my_ckpt, args, train_loader.dataset.num_classes)
    # 画图
    draw_attention(
        test_loader, adversary, args, unnormalize,
        teacher, finetune, retrain, renofeation, my_model
    )
