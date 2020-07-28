# [Renofeation: Improving the Adversarial Robustness of Transfer Learning via Noisy Feature Distillation](https://arxiv.org/abs/2002.02998)

## Don't fine-tune, Renofeate <img src="maintenance.png" width="30">

(Icon made by Eucalyp perfect from www.flaticon.com)

In our recent paper "[Improving the Adversarial Robustness of Transfer Learning via Noisy Feature Distillation](https://arxiv.org/abs/2002.02998)", we show that numerous fine-tuning methods are vulnerable to [adversarial examples based on the pre-trained model](https://openreview.net/forum?id=BylVcTNtDS). This poses security concerns for indutrial applications that are based on fine-tuning such as [Google's Cloud AutoML](https://cloud.google.com/automl) and [Microsoft's Custom Vision](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/).

To combat such attacks, we propose _**Re**-training with **no**isy **fea**ture distilla**tion**_ or Renofeation. Renofeation does not start training from pre-trained weights but rather re-initialize the weights and train with noisy feature distillation. To instantiate noisy feature distillation, we incorporate [spatial dropout](https://arxiv.org/abs/1411.4280) and [stochastic weight averaging](https://arxiv.org/abs/1803.05407) with feature distillation to avoid over-fitting to the pre-trained feature without hurting the generalization performance, which in turn improves the robustness.

To this end, we demonstrate empirically that the attack success rate can be reduced from 74.3%, 65.7%, 70.3%, and 50.75% down to 6.9%, 4.4%, 4.9%, and 6.9% for ResNet-18, ResNet-50, ResNet-101, and MobileNetV2, respectively. Moreover, the clean-data performance is comparable to the fine-tuning baseline!

For more details and an ablation study of our proposed method, please check out our [paper]()!


## Dependency

- PyTorch 1.0.0
- TorchVision 0.4.0
- AdverTorch 0.2.0

## Preparing datasets

### [Caltech-UCSD 200 Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html)
Layout should be the following for the dataloader to load correctly

```
CUB_200_2011/
|    README
|    bounding_boxes.txt
|    classes.txt
|    image_class_labels.txt
|    images.txt
|    train_test_split.txt
|--- attributes
|--- images/
|--- parts/
|--- train/
|--- test/
```

### [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
```
Flower_102/
|    imagelabels.mat
|    setid.mat
|--- jpg/
```

### [Stanford 120 Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
```
stanford_dog/
|    file_list.mat
|    test_list.mat
|    train_list.mat
|--- train/
|--- test/
|--- Images/
|--- Annotation/
```

### [Stanford 40 Actions](http://vision.stanford.edu/Datasets/40actions.html)
```
stanford_40/
|    attributes.txt
|--- ImageSplits/
|--- JPEGImages/
|--- MatlabAnnotations/
|--- XMLAnnotations/
```

### [MIT 67 Indoor Scenes](http://web.mit.edu/torralba/www/indoor.html)
```
MIT_67/
|    TrainImages.txt
|    TestImages.txt
|--- Annotations/
|--- Images/
|--- test/
|--- train/
```

### [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
```
GTSRB/
|    Readme-Images-Final-test.txt
|    Readme-Images.txt
|--- Final_Test/
|--- Final_Training/
```

## Model training and evaluation

**Be sure to modify the data path for each datasets for successful runs**

The scripts for training and evaluating *DELTA*, *Renofeation*, and *Re-training* are under `scripts/`.


## Citation

```
@article{chin2020renofeation,
  title={Improving the Adversarial Robustness of Transfer Learning via Noisy Feature Distillation},
  author={Chin, Ting-Wu and Zhang, Cha and Marculescu, Diana},
  journal={arXiv preprint arXiv:2002.02998},
  year={2020}
}
```

## To 锡涵 20200728
下一步工作主要分三块，按照这个顺序依次做就行：
- 画一些针对交通标志识别GTSRB的attention map
- 尝试poison attack在迁移学习场景下的效果
- 在VGG Face (pre-trained dataset)和PubFig上跑一下结果（难度比较大）


### 针对交通标志的attention map
attention map的一张示例图在backdoor/results/attention_demo里面，其中红色的部分代表关注度高，蓝色代表关注度低。这张示例图是对比正常样本和对抗样本的，左边一列是正常样本，右边一列是对抗样本，然后每一行是针对一个模型生成的。

对抗样本对比attention的生成代码在fineprune/plot/draw_attention.py中，我已经添加了注释。运行脚本在examples/plot/draw_attention.sh中。你要做的是根据这份代码写一份针对backdoor的可视化代码。具体要做的改动包括以下几点：
- 把对抗样本的输入替换为triggered输入，这里你可以直接使用两个dataset
```
for normal_batch, trigger_batch in zip(clean_test_dataset, triggered_test_dataset):
  ...
```
- 更改预测模型的数量。原来使用了四种模型，你换成三种模型：baseline，mag和divmag

原代码中一些注意细节
- 这里batch_size取的是1为了方便实现
- attention_layer只取了一个layer4，其他的layer都不考虑了

然后你要选出一些比较有代表性的图片，要满足以下要求：
- 所有模型在normal input的关注点都在交通标志上
- 在baseline和mag模型上，trigger input的attention map是关注在trigger上
- divmag在trigger input的关注点仍然在交通标志上

把挑选出来的图片保存起来，记录一下是test dataset的第几张图片

### 测试一下poison attack的效果
Poison attack 是通过在训练数据中掺杂有错误label的数据造成模型最终结果很差的一种攻击方法，最终模型在test数据集上准确率越低代表攻击效果越好。我觉得可以简单尝试一下这种攻击在迁移学习里面是不是会保留。

具体的方法跟我们之前backdoor的流程差不多，唯一的区别是在训练的时候backdoor使用带有trigger的错误数据，而poison使用没有trigger的错误数据。也就是说你只需要把原来的代码加trigger的部分去掉就可以。这个实验只需要在GTSRB上做一下，看看把20%的training data的label修改以后会不会影响迁移后模型的准确率。

这个实验实现起来比较简单，可以快速做一下。不过注意不要跟backdoor的代码混了，backdoor是更重要的。

### 在VGG Face上跑一下结果
VGG Face是一个类似ImageNet的大数据集，PubFig是小数据集，这个实验就是想在这对数据集上跑一下结果。也是使用网上公开的pretrain model先植入后门，然后再finetune一遍看攻击的效果。因为现在VGG Face公开的的模型都是ResNet50，没有ResNet18的，所以在这个场景里面我们的模型也有变一下使用ResNet50.

关于VGG Face的一个repo是[VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch)，代码可以参考。里面有一个resnet50_ft的预训练模型下载链接，从这里下载就可以，。不过要注意一下它的模型是保存为pkl格式，与我们之前保存的pt格式不一样。如何读取这个模型在[这里](https://github.com/cydonia999/VGGFace2-pytorch/blob/c6e10f277b31b972c78fac68a40464a36a46a10d/utils.py#L9)。

要做的包括以下几点：
- 先试试我们的方法在ResNet50上的效果。具体是使用ResNet50在GTSRB上跑一下结果（只跑一个数据集就够了），看看我们的方法是不是仍旧比mag要好。
- 然后从网上下载VGG Face ResNet50的pretrained model。这里你可以使用VGGFace2-pytorch跑一下模型的正确性，不过就要先下好VGG Face的数据集，可能比较大最好提前几天下好。
- 下载PubFig并写一个针对PubFig的dataset
- 尝试在PubFig上finetune看看结果

这里注意一点：VGG Face的input normalization跟我们之前使用的是不一样的，所以之前代码中的normalization要换掉。我建议你新建一个py_qianyi.py然后在那里面改，这个文件就是针对face的脚本了。

这个可能难度比较大先尝试一下，如果跑不出来也没关系。




## To 锡涵 20200713

### 针对迁移的后门攻击概述

下一步我们要尝试的解决方法是在训练（埋设）后门的时候只更新网络中权重最大的参数，不改变网络中权重小的参数。
这个想法的出发点是在第二步迁移的过程中权重较大的参数被改变的程度应该会小一点，这样埋设的后门有更大的可能保留下来。

### 我对backdoor的修改

我对backdoor文件夹做了以下几个修改：
- 添加prune.py，里面有一个函数weight_prune，它的作用是对网络所有的weight进行一个排序，并且把权重较大的weight进行标记。标记结果保存在module.mask中（mask是一个01矩阵，大小与module.weight一样），这里module指网络中的所有conv层。这里权重的数量由weight_prune中的ratio控制。
- 把fineprune/finetuner.py复制为attack_finetuner.py，在训练GTSRB的过程中就使用这个attack_finetuner.py。
- 修改了attack_finetuner.py中的379-385行，根据conv层的mask更新梯度，这样权重较小的weight的grad会被清零，在后面optimizer.step时也就不会被更新。
- py_qianyi.py的396-404行使用AttackFinetuner，405-410使用正常的Finetuner。
在训练GTSRB（也就是埋设后门的时候），使用AttackFinetuner（也就是要把py_qianyi.py的method参数设为backdoor_finetune）。
而在进行第二次迁移（即MIT67以及CUB200），使用正常的FInetuner，也就是不用设置py_qianyi.py的method参数。
  - py_qianyi.py的123-125行，添加一个参数backdoor_update_ratio控制更新大权重weight的比例。
如果backdoor_update_ratio是0.8，那么网络中权重最小的80%的weight就不会被更新，权重最大的20%的weight会被更新。
  - py_qianyi.py的397-399行，调用weight_prune为student标记所有conv层可以更新的mask

### 你要做的

- 把我添加的这些地方看懂，尤其是weight_prune函数和attack_finetuner.py中的379-385行，有不明白的地方可以问我
- 我只是大概把关键的地方写了，并没有运行过，里面应该会有一些bug，你根据这些更新的部分把bug改一下，把代码跑通。
- 跑几组实验，把backdoor_update_ratio设为0.5, 0.7, 0.9，迁移到GTSRB的同时埋设backdoor，然后看看在MIT67和CUB200两个数据集上攻击的效果。

### 一些细节及可能不太理解的地方

- 现在两个阶段的迁移我们使用不同的代码。对GTSRB的迁移（埋设backdoor）使用attack_finetuner.py中的AttackFinetuner，对MIT67和CUB200的迁移还按照原来的方法使用Finetuner。对GTSRB迁移时，你在运行py_qianyi.py的时候要添加method参数，即``--method backdoor_finetune''，并设置backdoor_update_ratio。

- 参数backdoor_update_ratio是控制有多少比例的小weight权重在更新时fix住。如果backdoor_update_ratio=0.8，代表在埋设backdoor时，网络中权重最小的80%的weight是不更新的，只有权重最大的20%的weight会更新。



## To 锡涵
我们现在使用这份代码进行迁移学习的实验，现在的代码与之前的不同点在于输入图片规模。现在我们在真实图片上做实验（224x224，以前的CIFAR都是32x32的规模），使用的预训练模型是ImageNet pre-trained model，代码会自动从pytorch官网上下载。我现在正在实现对抗样本的部分，你的任务是在这份代码上实现backdoor的baseline。

examples/backdoor/finetune.sh 是迁移学习的代码，现在迁移的目标数据集是MIT Indoor 67，你可以先把这个脚本跑一下。

### 代码结构
- 主要涉及到的代码在fineprune文件夹下，有两个python文件: finetune.py和finetuner.py。
- finetune.py是一个入口文件
  - 138-167是数据准备，对你比较重要。
  - 178-183行是读取teacher model，这里是ImageNet pre-trained mode会自动下载，不需要像我们之前一样自己训练一个teacher model。
- finetuner.py是具体的训练代码需要仔细看一下，但是其中涉及l2sp, feat_lmda以及SWA的你可以不用管，是各种trick你用不到，重点是理清楚整个训练的流程。
- 其他相关代码包括utils.py, eval_robustness.py, 这两个知道会用到就可以，不用仔细研究。datasets下的文件是对不同数据集的读取文件。

### 任务
在backdoor这块我们需要先实现在一个新的数据集上进行迁移学习的baseline，数据集叫German Traffic Sign Recognition Benchmark (GTSRB) 是一个德国交通标志检测数据。

#### 下载数据集
数据集网站在http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset，可以点击下载数据集。你在服务器是可以通过以下命令，然后解压放到data文件夹下即可。
```
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
```

#### 实现新数据集的dataset
- 在dataset文件夹下新建一个gtsrb.py，里面实现一个新类叫GTSRBData，你可以参考dataset/mit67.py下的实现。关于读取GTSRB的数据可以参考[gtsrb_loader.py](https://github.com/abursuc/dldiy-gtsrb/blob/master/loaders/gtsrb_loader.py)。这一步你可以理解为把gtsrb_loader.py重新包装一下，使得它的数据集格式与mit67.py相同。
- 这里还有一个细节是输入图片的转换问题（transform），我们这里有统一的转换规则，在backdoor/finetune.py中的134-156行，就不要使用gtsrb_loader.py中的转换方法了。
- 实现后你可以把GTSRBData加到finetune.py中，然后在finetune.py的157行（读取train_set和test_set之后）设置一个断点check一下
  - 检查train_set和test_set的长度是否合理
```
len(train_set)
```
  - 检查train_set图片的范围是否合理，即GTSRBData一张图片的min、max应该与MIT67一张图片的minmax相同
```
min(train_set[0]), max(train_set[0])
```

#### 在GTSRBData上finetune
实现数据集后你可以尝试finetune一下，看看准确率怎么样。其中learning rate 可以尝试两种设置，分别是5e-3和1e-2，脚本与examples/backdoor/finetune.sh一样，只不过改一下数据集即可。
