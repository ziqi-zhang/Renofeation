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
# total_ratio=0.55
# init_ratio=0.3
# per_ratio=0.1
# interval=10

# all init
total_ratio=0.6
init_ratio=0.6
per_ratio=0.1
interval=10000

# # avg
# total_ratio=0.9
# init_ratio=0
# per_ratio=0.05
# interval=400

# # front
# total_ratio=0.9
# init_ratio=0
# per_ratio=0.1
# interval=500

for i in 0 
do
    for ratio in 0.8 0.9
    do
        total_ratio=${ratio}
        init_ratio=${ratio}
        per_ratio=0
        interval=10000

        DATASET=${DATASETS[i]}
        DATASET_NAME=${DATASET_NAMES[i]}
        DATASET_ABBR=${DATASET_ABBRS[i]}

        NAME=resnet18_${DATASET_ABBR}_constlr_\
total${total_ratio}_init${init_ratio}_per${per_ratio}_int${interval}_trainall_\
lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
        DIR=results/fineprune/perlayer_weight/constlr

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
        --adv_test_interval -1 \
        --feat_layers ${layer} \
        --momentum ${mmt} \
        --output_dir ${DIR} \
        --method perlayer_weight \
        --prune_interval $interval \
        --weight_total_ratio $total_ratio \
        --weight_ratio_per_prune $per_ratio \
        --weight_init_prune_ratio $init_ratio \
        --train_all \
        --const_lr \
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