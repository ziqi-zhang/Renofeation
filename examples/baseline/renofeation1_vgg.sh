#!/bin/bash


iter=30000
id=0
splmda=0
layer=12345
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

    NAME=vgg11_${DATASET_ABBR}_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
    DIR=results/baseline/vgg/renofeation1

    CUDA_VISIBLE_DEVICES=$1 \
    python -u init_fd_train.py \
    --iterations ${iter} \
    --datapath data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
    --name ${NAME} \
    --batch_size 32 \
    --feat_lmda ${lmda} \
    --lr ${lr} \
    --network vgg11_bn \
    --weight_decay ${wd} \
    --beta 1e-2 \
    --test_interval 100000 \
    --feat_layers ${layer} \
    --momentum ${mmt} \
    --reinit \
    --dropout 1e-1 \
    --output_dir ${DIR} \
    --adv_test_interval -1 \
    # &


    # WORKDIR=${DIR}/${NAME}
    # python -u eval_robustness.py \
    # --datapath data/${DATASET}/ \
    # --dataset ${DATASET_NAME} \
    # --network mbnetv2 \
    # --checkpoint ${WORKDIR}/ckpt.pth \
    # --batch_size 128 \
    # > ${WORKDIR}/eval.log



done