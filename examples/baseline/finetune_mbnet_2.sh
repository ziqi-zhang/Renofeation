#!/bin/bash


iter=10000
id=1
splmda=0
lmda=0
layer=1234
lr=5e-3
wd=1e-4
mmt=0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog VisDA)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data VisDaDATA)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120 visda)

for i in 1 2
do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    NAME=mbnetv2_${DATASET_ABBR}_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
    DIR=results/baseline/mbnetv2/finetune_2

    # CUDA_VISIBLE_DEVICES=0 \
    # python -u train.py \
    # --iterations ${iter} \
    # --datapath data/${DATASET}/ \
    # --dataset ${DATASET_NAME} \
    # --name $NAME \
    # --batch_size 64 \
    # --feat_lmda ${lmda} \
    # --lr ${lr} \
    # --network mbnetv2 \
    # --weight_decay ${wd} \
    # --beta 1e-2 \
    # --test_interval 1000 \
    # --feat_layers ${layer} \
    # --momentum ${mmt} \
    # --output_dir $DIR \
    # # &


    WORKDIR=${DIR}/${NAME}
    python -u eval_robustness.py \
    --datapath data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
    --network mbnetv2 \
    --checkpoint ${WORKDIR}/ckpt.pth \
    --batch_size 32 \
    > ${WORKDIR}/eval.log

done