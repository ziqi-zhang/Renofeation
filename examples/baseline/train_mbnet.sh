#!/bin/bash


iter=90000
id=1
splmda=0
lmda=0
layer=12345
lr=1e-2
wd=5e-3
mmt=0.9

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog VisDA)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data VisDaDATA)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120 visda)

for i in 0 1 2 3 4
do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    DIR=results/baseline/mbnetv2/retrain
    NAME=mbnetv2_${DATASET_ABBR}_reinit_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}

    # CUDA_VISIBLE_DEVICES=$1 \
    # python -u train.py \
    # --iterations ${iter} \
    # --datapath data/${DATASET}/ \
    # --dataset ${DATASET_NAME} \
    # --name ${NAME} \
    # --batch_size 64 \
    # --feat_lmda ${lmda} \
    # --lr ${lr} \
    # --network mbnetv2 \
    # --weight_decay ${wd} \
    # --beta 1e-2 \
    # --test_interval 10000 \
    # --feat_layers ${layer} \
    # --momentum ${mmt} \
    # --reinit \
    # --output_dir ${DIR} \
    # # &



    WORKDIR=${DIR}/${NAME}
    CUDA_VISIBLE_DEVICES=$1 \
    python -u eval_robustness.py \
    --datapath data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
    --network mbnetv2 \
    --checkpoint ${WORKDIR}/ckpt.pth \
    --batch_size 32 \
    > ${WORKDIR}/eval.log



done