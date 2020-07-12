#!/bin/bash


iter=30000
id=1
splmda=0
layer=1234
lr=5e-3
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

    DIR=results/baseline/renofeation2
    NAME=resnet18_${DATASET_ABBR}_reinit_constlr_lr${lr}_do1e-1_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
    CKPT_DIR=results/baseline/renofeation1/resnet18_${DATASET_ABBR}_reinit_lr1e-2_iter90000_feat5e0_wd5e-3_mmt0.9_0


    CUDA_VISIBLE_DEVICES=0 \
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
    --adv_test_interval -1 \
    --feat_layers ${layer} \
    --momentum ${mmt} \
    --dropout 1e-1 \
    --output_dir ${DIR} \
    --swa --swa_freq 500 --swa_start 0 \
    --const_lr \
    --checkpoint ${CKPT_DIR}/ckpt.pth \
    # &


    # WORKDIR=${DIR}/${NAME}
    # python -u eval_robustness.py \
    # --datapath data/${DATASET}/ \
    # --dataset ${DATASET_NAME} \
    # --network resnet18 \
    # --checkpoint ${WORKDIR}/ckpt.pth \
    # > ${WORKDIR}/eval.log



done