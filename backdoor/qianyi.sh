#!/bin/bash

iter=10000
id=1
splmda=0
lmda=0
layer=1234
mmt=0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog GTSRB)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data GTSRBData)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120 gtsrb)
LEARNING_RATE=5e-3
WEIGHT_DECAY=1e-4
#PORTION=(0.98 0.2 0.5 0.8 0.9 0.95 0.96 0.97 0.99 1)

for i in 4
do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}
    lr=${LEARNING_RATE}
    wd=${WEIGHT_DECAY}
    #portion=${PORTION[i]}

    NAME=resnet18_${DATASET_ABBR}_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
    DIR=results/qianyi/

    CUDA_VISIBLE_DEVICES=1 \
    python -u py_qianyi.py \
    --student_datapath ../data/${DATASET} \
    --student_dataset ${DATASET_NAME} \
    --iterations ${iter} \
    --name ${NAME} \
    --batch_size 64 \
    --feat_lmda ${lmda} \
    --lr ${lr} \
    --network resnet18 \
    --weight_decay ${wd} \
    --beta 1e-2 \
    --test_interval 1000 \
    --adv_test_interval -1 \
    --feat_layers ${layer} \
    --momentum ${mmt} \
    --output_dir ${DIR} \
    --const_lr \
    --checkpoint ../examples/backdoor/results/backdoor/trigger/resnet18_gtsrb_lr5e-3_iter10000_feat0_wd1e-4_mmt0_1_0.2/ckpt.pth \
    #--student_ckpt results/qianyi/resnet18_mit67_lr5e-3_iter10000_feat0_wd1e-4_mmt0_1/ckpt.pth

done

