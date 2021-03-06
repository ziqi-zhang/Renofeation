#!/bin/bash
export PYTHONPATH=../..:$PYTHONPATH

iter=10000
id=1
splmda=0
lmda=0
layer=1234
lr=5e-3
wd=1e-4
mmt=0
# wd=0
# mmt=0.9

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120)

for i in 0 
do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    NAME=vgg11_${DATASET_ABBR}_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
    DIR=results/baseline/vgg/finetune

    # CUDA_VISIBLE_DEVICES=$1 \
    # python -u fineprune/finetune.py \
    # --iterations ${iter} \
    # --datapath data/${DATASET}/ \
    # --dataset ${DATASET_NAME} \
    # --name ${NAME} \
    # --batch_size 64 \
    # --feat_lmda ${lmda} \
    # --lr ${lr} \
    # --network vgg11_bn \
    # --weight_decay ${wd} \
    # --beta 1e-2 \
    # --test_interval 10000 \
    # --adv_test_interval -1 \
    # --feat_layers ${layer} \
    # --momentum ${mmt} \
    # --output_dir ${DIR} \
    # &


    WORKDIR=${DIR}/${NAME}
    CUDA_VISIBLE_DEVICES=$1 \
    python -u eval_robustness.py \
    --datapath data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
    --network vgg11_bn \
    --checkpoint ${WORKDIR}/ckpt.pth \
    --batch_size 8 \
    > ${WORKDIR}/eval.log

done