#!/bin/bash

iter=10000
id=1
splmda=0
lmda=0
layer=1234
mmt=0

DATASETS=(MIT_67 GTSRB)
DATASET_NAMES=(MIT67Data GTSRBData)
DATASET_ABBRS=(mit67 gtsrb)
LEARNING_RATE=(5e-3 1e-2)
WEIGHT_DECAY=(1e-4 5e-3)

for i in 1
do
    for j in 0
    do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}
    lr=${LEARNING_RATE[j]}
    wd=${WEIGHT_DECAY[j]}

    NAME=resnet18_${DATASET_ABBR}_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
    DIR=results/backdoor/finetune/

    CUDA_VISIBLE_DEVICES=$j \
    python -u ../../backdoor/jiexu_finetune.py \
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

    done
done
