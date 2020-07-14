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

DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120)

# test
total_num=500
init_num=$total_num
per_num=0
interval=${iter}


for i in 0
do
    
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    NAME=resnet18_${DATASET_ABBR}_\
total${total_num}_init${init_num}_per${per_num}_int${interval}_\
lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
    CKPT_DIR=results/baseline/finetune/resnet18_mit67_lr5e-3_iter10000_feat0_wd1e-4_mmt0_1
    DIR=results/fineprune/taylorfilter/pretrain
    # DIR=results/test

    CUDA_VISIBLE_DEVICES=0 \
    python -u fineprune/finetune.py \
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
    --output_dir ${DIR} \
    --method taylor_filter \
    --prune_interval $interval \
    --filter_total_number $total_num \
    --filter_number_per_prune $per_num \
    --filter_init_prune_number $init_num \
    # &


    # WORKDIR=${DIR}/${NAME}
    # python -u eval_robustness.py \
    # --datapath data/${DATASET}/ \
    # --dataset ${DATASET_NAME} \
    # --network resnet18 \
    # --checkpoint ${WORKDIR}/ckpt.pth \
    # --batch_size 8 \
    # > ${WORKDIR}/eval.log

done