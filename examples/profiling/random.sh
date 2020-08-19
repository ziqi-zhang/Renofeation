#!/bin/bash
export PYTHONPATH=../..:$PYTHONPATH

iter=30000
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

low_bound=0
for i in 0
do
    for ratio in 0.1 0.3 0.5
    do
        for trial_iter in  3000
        do
            total_ratio=${ratio}
            init_ratio=${ratio}
            per_ratio=0.1
            interval=10000

            DATASET=${DATASETS[i]}
            DATASET_NAME=${DATASET_NAMES[i]}
            DATASET_ABBR=${DATASET_ABBRS[i]}

            NAME=resnet18_${DATASET_ABBR}_do_\
total${total_ratio}_init${init_ratio}_per${per_ratio}_int${interval}_trainall_\
lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
            DIR=results/profiling/random

            CUDA_VISIBLE_DEVICES=$1 \
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
            --test_interval 30000 \
            --adv_test_interval -1 \
            --feat_layers ${layer} \
            --momentum ${mmt} \
            --output_dir ${DIR} \
            --method random \
            --prune_interval $interval \
            --weight_total_ratio $total_ratio \
            --weight_ratio_per_prune $per_ratio \
            --weight_init_prune_ratio $init_ratio \
            --train_all \
            --weight_low_bound $low_bound \
            --trial_iteration ${trial_iter} \
            --trial_lr ${lr} \
            --trial_momentum 0.9 \
            --trial_weight_decay 0 \
            &


            # WORKDIR=${DIR}/${NAME}
            # python -u eval_robustness.py \
            # --datapath data/${DATASET}/ \
            # --dataset ${DATASET_NAME} \
            # --network resnet18 \
            # --checkpoint ${WORKDIR}/ckpt.pth \
            # --batch_size 8 \
            # > ${WORKDIR}/eval.log

        done
    done
done