import torch
import torch.nn as nn

import numpy as np
import time
from pdb import set_trace as st

def weight_prune(
    model,
    begin_layer,
    prune_ratio,
    model_layer_depth,
    kaiming_init=False,
):
    total = 0
    for name, module in model.named_modules():
        if ( isinstance(module, nn.Conv2d) and 
            model_layer_depth[name] > model_layer_depth[begin_layer]
        ):
                total += module.weight.data.numel()
    
    conv_weights = torch.zeros(total)
    index = 0
    for name, module in model.named_modules():
        if ( isinstance(module, nn.Conv2d) and 
            model_layer_depth[name] > model_layer_depth[begin_layer]
        ):
            size = module.weight.data.numel()
            conv_weights[index:(index+size)] = module.weight.data.view(-1).abs().clone()
            index += size
    
    y, i = torch.sort(conv_weights)
    thre_index = int(total * prune_ratio)
    thre = y[thre_index]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    
    for name, module in model.named_modules():
        if ( isinstance(module, nn.Conv2d) and 
            model_layer_depth[name] > model_layer_depth[begin_layer]
        ):
            weight_copy = module.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float()
            pruned = pruned + mask.numel() - torch.sum(mask)
            # np.random.shuffle(mask)
            module.weight.data.mul_(mask)
            
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            remain_ratio = int(torch.sum(mask)) / mask.numel()
            print('layer {:s} \t total params: {:d} \t remaining params: {:d}({:.2f})'.
                format(name, mask.numel(), int(torch.sum(mask)), remain_ratio))
            
    if zero_flag:
        raise RuntimeError("There exists a layer with 0 parameters left.")
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    return model


def weight_prune_all(
    model,
    prune_ratio,
    random_prune=False,
    mode="small",
    kaiming_init=False,
):
    total = 0
    for name, module in model.named_modules():
        if ( isinstance(module, nn.Conv2d) ):
                total += module.weight.data.numel()
    
    conv_weights = torch.zeros(total)
    index = 0
    for name, module in model.named_modules():
        if ( isinstance(module, nn.Conv2d) ):
            size = module.weight.data.numel()
            conv_weights[index:(index+size)] = module.weight.data.view(-1).abs().clone()
            index += size
    
    y, i = torch.sort(conv_weights)
    # thre_index = int(total * prune_ratio)
    # thre = y[thre_index]
    if mode == "small":
        thre_index = int(total * prune_ratio)
        thre = y[thre_index]
        print('Pruning threshold: {}'.format(thre))
    elif mode == "large":
        thre_index = int(total * prune_ratio)
        thre = y[-thre_index]
        print('Pruning threshold: {}'.format(thre))
    elif mode == "mid":
        thre_index = int(total * prune_ratio / 2)
        low_thre = y[thre_index]
        high_thre = y[-thre_index]
        print('Pruning threshold: {} to {}'.format(low_thre, high_thre))
    else:
        raise NotImplementedError

    pruned = 0
    
    zero_flag = False
    
    for name, module in model.named_modules():
        if ( isinstance(module, nn.Conv2d) ):
            weight_copy = module.weight.data.abs().clone()
            # mask = weight_copy.gt(thre).float()
            if mode == "small":
                mask = weight_copy.gt(thre).float()
            elif mode == "large":
                mask = weight_copy.lt(thre).float()
            elif mode == "mid":
                mask = weight_copy.gt(low_thre).float() * weight_copy.lt(high_thre).float()
            if random_prune:
                print(f"Random prune {name}")
                mask = np.zeros(weight_copy.numel()) + 1
                prune_number = round(prune_ratio * weight_copy.numel())
                mask[:prune_number] = 0
                np.random.shuffle(mask)
                mask = mask.reshape(weight_copy.shape)
                mask = torch.Tensor(mask)

            pruned = pruned + mask.numel() - torch.sum(mask)
            # np.random.shuffle(mask)
            module.weight.data.mul_(mask)
            if kaiming_init:
                init_weight = module.weight.data.clone()
                nn.init.kaiming_normal_(init_weight, mode='fan_out', nonlinearity='relu')
                init_weight.mul_(1-mask)
                # print(module.weight.data[0,0,0])
                # print(init_weight[0,0,0])
                module.weight.data.add_(init_weight)
                # print(module.weight.data[0,0,0])
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            remain_ratio = int(torch.sum(mask)) / mask.numel()
            print('layer {:s} \t total params: {:d} \t remaining params: {:d}({:.2f})'.
                format(name, mask.numel(), int(torch.sum(mask)), remain_ratio))
            
    if zero_flag:
        raise RuntimeError("There exists a layer with 0 parameters left.")
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))
    return model



def weight_train_step(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    epoch, 
    device,
    log_interval=10,
    fix_bn_dict=None,
):
    model.train()
    if fix_bn_dict is not None:
        for name, module in fix_bn_dict.items():
            module.eval()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        #-----------------------------------------
        for k, m in enumerate(model.modules()):
            # print(k, m)
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().to(device)
                m.weight.grad.data.mul_(mask)
            if isinstance(m, nn.Linear):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().to(device)
                m.weight.grad.data.mul_(mask)
        #-----------------------------------------
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total
    
        if (batch_idx + 1) % log_interval == 0:
            print(
                f"[WeightTrain] Epoch: {epoch} "
                f"[{batch_idx * len(inputs)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item():.2f}\t"
                f"Accuracy: {acc:.1f} ({correct}/{total})"
            )

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc
