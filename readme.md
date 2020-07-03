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
