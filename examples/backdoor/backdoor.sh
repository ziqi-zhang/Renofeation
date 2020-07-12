#!/bin/bash

iter=10000
id=1
splmda=0
lmda=0
layer=1234
mmt=0

DATASETS=GTSRB
DATASET_NAMES=GTSRBData
DATASET_ABBRS=gtsrb
LEARNING_RATE=5e-3
WEIGHT_DECAY=1e-4
PORTION=(0.98 0.2 0.5 0.8 0.9 0.95 0.96 0.97 0.99 1)

for i in 0 
do
    DATASET=${DATASETS}
    DATASET_NAME=${DATASET_NAMES}
    DATASET_ABBR=${DATASET_ABBRS}
    lr=${LEARNING_RATE}
    wd=${WEIGHT_DECAY}
    portion=${PORTION[i]}

    NAME=resnet18_${DATASET_ABBR}_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}_${portion}
    DIR=../../backdoor/results/trigger/

    CUDA_VISIBLE_DEVICES=1 \
    python -u ../../backdoor/trigger.py \
    --iterations ${iter} \
    --datapath ../../data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
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
    --argportion $portion \
    &

done
