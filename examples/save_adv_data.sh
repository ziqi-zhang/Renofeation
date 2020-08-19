#!/bin/bash


iter=5000
id=0
splmda=0
layer=1234
lr=1e-2
wd=5e-3
mmt=0.9
lmda=5e0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120)

for i in 0 
do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    DIR=results/advdata/

    python -u save_adv_data.py \
    --datapath data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
    --network resnet18 \
    --save_dir ${DIR} \


done