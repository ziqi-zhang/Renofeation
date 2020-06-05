#!/bin/bash


iter=90000
id=1
splmda=0
lmda=0
layer=1234
lr=1e-2
wd=5e-3
mmt=0.9

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120)

for i in 0 
do
    for RATIO in 0.3 0.5 0.7
    do

        DATASET=${DATASETS[i]}
        DATASET_NAME=${DATASET_NAMES[i]}
        DATASET_ABBR=${DATASET_ABBRS[i]}

        CUDA_VISIBLE_DEVICES=2 \
        python -u prune.py \
        --iterations ${iter} \
        --datapath data/${DATASET}/ \
        --dataset ${DATASET_NAME} \
        --name resnet18_${DATASET_ABBR}_lr${lr}_iter${iter}_${id} \
        --batch_size 64 \
        --feat_lmda ${lmda} \
        --lr ${lr} \
        --network resnet18 \
        --weight_decay ${wd} \
        --beta 1e-2 \
        --test_interval 1000 \
        --feat_layers ${layer} \
        --momentum ${mmt} \
        --output_dir results/prune \
        --ratio $RATIO \
        &
        # --log \
        # &

        # exit

    done
done