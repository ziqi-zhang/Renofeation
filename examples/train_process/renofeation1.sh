#!/bin/bash


iter=90000
id=1
splmda=0
layer=1234
lr=1e-2
wd=5e-3
mmt=0.9
lmda=5e0

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120)

for i in 2
do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    DIR=results/train_process/renofeation1
    # DIR=results/test
    NAME=resnet18_${DATASET_ABBR}_reinit_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}

    CUDA_VISIBLE_DEVICES=$1 \
    python -u init_fd_train.py \
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
    --dropout 1e-1 \
    --output_dir ${DIR} \
    --adv_test_interval 5000 \
    # &


    # WORKDIR=${DIR}/${NAME}
    # python -u eval_robustness.py \
    # --datapath data/${DATASET}/ \
    # --dataset ${DATASET_NAME} \
    # --network resnet18 \
    # --checkpoint ${WORKDIR}/ckpt.pth \
    # --batch_size 128 \
    # > ${WORKDIR}/eval.log



done