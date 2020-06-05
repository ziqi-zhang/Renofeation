import torch
import torch.nn as nn
import math

from itertools import chain
from pdb import set_trace as st

resnet18_layers = [
    "input",
    "conv1",
    "bn1",
    
]
for layer_id in [1,2,3,4]:
    for block_id in range(2):
        block_names = [
            f"layer{layer_id}.{block_id}.conv1",
            f"layer{layer_id}.{block_id}.bn1",
            f"layer{layer_id}.{block_id}.conv2",
            f"layer{layer_id}.{block_id}.bn2",
        ]
        if (layer_id > 1 and block_id == 0):
            block_names.append(f"layer{layer_id}.{block_id}.downsample.0")
        resnet18_layers += block_names
resnet18_layers.append("last_linear")
# for layer in resnet56_layers:
#     print(layer)
resnet18_layer_depth = {}
for i, name in enumerate(resnet18_layers):
    resnet18_layer_depth[name] = i+1
# for layer in resnet18_layer_depth:
#     print(layer, resnet18_layer_depth[layer])
# st()

def init_resnet18(
    model, 
    num_classes,
    begin_layer,
    lr,
    train_all=False,
    random_prune = False,
):
    in_feat = model.last_linear.in_features
    model.last_linear = nn.Linear(in_feat, num_classes)
    param_group = [
        {
            'params':model.last_linear.parameters(),
            'lr': lr*10,
        }
    ]
    model_params_list = []
    fix_bn_dict = {}
    if train_all:
        for name, module in model.named_modules():
            if name == "last_linear":
                continue
            if name in resnet18_layer_depth.keys():
                param_group += [
                    {
                        'params': module.parameters(),
                        'lr': lr,
                    }
                ]
                print(f"Train all add {name}")
    else:
        assert begin_layer != "fc"
        for name, module in model.named_modules():
            if name == "last_linear":
                continue
            if name in resnet18_layer_depth.keys():
                if resnet18_layer_depth[name] > resnet18_layer_depth[begin_layer]:
                    # model_params_list.append(module.parameters())
                    param_group += [
                        {
                            'params': module.parameters(),
                            'lr': lr,
                        }
                    ]
                    print(f"Add {name}")
                elif isinstance(module, nn.BatchNorm2d):
                    fix_bn_dict[name] = module
    model_params = chain(*model_params_list)
        
    if random_prune:
        model = randomize_resnet18(model, begin_layer)
        
    return model, param_group, fix_bn_dict

def randomize_resnet18(
    model, begin_layer,
):
    for name, module in model.named_modules():
        if name in resnet18_layer_depth.keys() and resnet18_layer_depth[name] > resnet18_layer_depth[begin_layer]:
            print(f"Randomize {name} weights")
            m = module
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    return model

resnet50_layers = [
    "conv1",
    "bn1",
    
]
for layer_id, block_num in zip([1,2,3,4], [3,3,5,3]):
    for block_id in range(block_num):
        block_names = [
            f"layer{layer_id}.{block_id}.conv1",
            f"layer{layer_id}.{block_id}.bn1",
            f"layer{layer_id}.{block_id}.conv2",
            f"layer{layer_id}.{block_id}.bn2",
            f"layer{layer_id}.{block_id}.conv3",
            f"layer{layer_id}.{block_id}.bn3",
        ]
        if (layer_id > 1 and block_id == 0):
            block_names.append(f"layer{layer_id}.{block_id}.downsample.0")
            block_names.append(f"layer{layer_id}.{block_id}.downsample.1")
        resnet50_layers += block_names
resnet50_layers.append("last_linear")
# for layer in resnet50_layers:
#     print(layer)
resnet50_layer_depth = {}
for i, name in enumerate(resnet50_layers):
    resnet50_layer_depth[name] = i+1
# for layer in resnet50_layer_depth:
#     print(layer, resnet50_layer_depth[layer])
# st()

def init_resnet50(
    model, 
    num_classes,
    begin_layer,
    train_all=False,
):
    in_feat = model.last_linear.in_features
    model.last_linear = nn.Linear(in_feat, num_classes)
    if train_all:
        model_params = model.parameters()
    else:
        model_params_list = []
        fix_bn_dict = {}
        assert begin_layer != "fc"
        for name, module in model.named_modules():
            if name in resnet50_layer_depth.keys():
                if resnet50_layer_depth[name] > resnet50_layer_depth[begin_layer]:
                    model_params_list.append(module.parameters())
                    print(f"Add {name}")
                elif isinstance(module, nn.BatchNorm2d):
                    fix_bn_dict[name] = module
        model_params = chain(*model_params_list)
        
    return model, model_params, fix_bn_dict