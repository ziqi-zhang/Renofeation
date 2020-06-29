
import weight.in_resnet as reinit_in_resnet
import weight.in_vgg as reinit_in_vgg
import weight.weight as weight_prune

from model.fe_resnet import resnet18_dropout, resnet50_dropout, resnet101_dropout
import model as model_lib

from pdb import set_trace as st

import torchvision

def reinit_model( 
    begin_layer,
    model,
    num_classes,
    train_all,
    prune_ratio,
    lr,
    random_prune=False,
):
    
    if isinstance(model, torchvision.models.resnet.ResNet):
        if len(model.layer1) == 2:
            model, model_params, fix_bn_dict = reinit_in_resnet.init_resnet18(
                model, num_classes, begin_layer, lr, train_all,
                random_prune=random_prune,
            )
            model = weight_prune.weight_prune(
                model,
                begin_layer,
                prune_ratio,
                reinit_in_resnet.resnet18_layer_depth,
            )
            return model, model_params, fix_bn_dict
        elif len(model.layer1) == 3:
            raise RuntimeError("lr")
            return reinit_in_resnet.init_resnet50(
                model, num_classes, begin_layer, train_all,
            )
        else:
            raise NotImplementedError
    elif isinstance(model, torchvision.models.vgg.VGG):
        raise RuntimeError("lr")
        return reinit_in_vgg.init_vgg16(
            model, num_classes, begin_layer, train_all,
        )
    else:
        st()
        raise NotImplementedError

def prune( 
    model,
    prune_ratio,
    mode="small",
    kaiming_init=False,
):
    if isinstance(model, model_lib.fe_resnet.ResNet):
        if len(model.layer1) == 2:

            model = weight_prune.weight_prune_all(
                model,
                prune_ratio,
                mode=mode,
                kaiming_init=kaiming_init,
            )
            return model
        elif len(model.layer1) == 3:
            raise RuntimeError("lr")
            return reinit_in_resnet.init_resnet50(
                model, num_classes, begin_layer, train_all,
            )
        else:
            raise NotImplementedError
    elif isinstance(model, torchvision.models.vgg.VGG):
        raise RuntimeError("lr")
        return reinit_in_vgg.init_vgg16(
            model, num_classes, begin_layer, train_all,
        )
    else:
        st()
        raise NotImplementedError

if __name__=="__main__":
    st()