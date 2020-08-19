#!/bin/bash

iter=5000
id=1
splmda=0
lmda=0
layer=1234
lr=5e-3
wd=1e-4
mmt=0

DATASETS=(MIT_67 stanford_dog GTSRB LISA pubfig83 CUB_200_2011 Flower_102 stanford_40 stanford_dog GTSRB)
DATASET_NAMES=(MIT67Data SDog120Data GTSRBData LISAData PUBFIGData CUB200Data Flower102Data Stanford40Data SDog120Data GTSRBData)
DATASET_ABBRS=(mit67 sdog120 gtsrb lisa pubfig cub200 flower102 stanford40 sdog120 gtsrb)

for i in 2 
do

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    #NAME=resnet18_${DATASET_ABBR}_do_\
    #total${total_ratio}_init${init_ratio}_per${per_ratio}_int${interval}_trainall_\
    #lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
    # DIR=results/datasetgrad/global_datasetgrad_divmag/lrschedule/trial${trial_iter}
    DIR=results/plot/draw_attention
    NAME=lisa
    BASELINE_DIR=results/backdoor/baseline/fixed_${DATASET_ABBR}_0.0/TWO_ckpt.pth
    DIVMAG_DIR=results/backdoor/divmag/fixed_${DATASET_ABBR}_0.0/divmag_two_ckpt.pth
    MAG_DIR=results/backdoor/mag/fixed_${DATASET_ABBR}_0.0/weight_two_ckpt.pth

    CUDA_VISIBLE_DEVICES=0 \
    python -u draw_attention.py \
    --iterations ${iter} \
    --datapath ../data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
    --name ${NAME} \
    --batch_size 1 \
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
    --method weight \
    --train_all \
    --dropout 1e-1 \
    --baseline_ckpt $BASELINE_DIR \
    --mag_ckpt $MAG_DIR \
    --divmag_ckpt $DIVMAG_DIR \
    --fixed_pic \
    
done
