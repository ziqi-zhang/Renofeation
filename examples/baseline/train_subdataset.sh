#!/bin/bash


iter=30000
id=1
splmda=0
lmda=0
layer=1234
lr=1e-2
wd=5e-3
mmt=0.9

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog VisDA)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data VisDaDATA)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120 visda)

for i in 5
do
    for shots in 1000 2000 3000
    do
        DATASET=${DATASETS[i]}
        DATASET_NAME=${DATASET_NAMES[i]}
        DATASET_ABBR=${DATASET_ABBRS[i]}

        DIR=results/baseline/retrain_subdataset
        NAME=resnet18_${DATASET_ABBR}_${shots}_reinit_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}

        CUDA_VISIBLE_DEVICES=0 \
        python -u train.py \
        --iterations ${iter} \
        --datapath data/${DATASET}/ \
        --dataset ${DATASET_NAME} \
        --name ${NAME} \
        --batch_size 64 \
        --feat_lmda ${lmda} \
        --lr ${lr} \
        --network resnet18 \
        --weight_decay ${wd} \
        --beta 1e-2 \
        --test_interval 1000 \
        --feat_layers ${layer} \
        --momentum ${mmt} \
        --reinit \
        --output_dir ${DIR} \
        --shot ${shots} \
        &



        # WORKDIR=${DIR}/${NAME}
        # python -u eval_robustness.py \
        # --datapath data/${DATASET}/ \
        # --dataset ${DATASET_NAME} \
        # --network resnet18 \
        # --checkpoint ${WORKDIR}/ckpt.pth \
        # --batch_size 64 \
        # > ${WORKDIR}/eval.log


    done
done