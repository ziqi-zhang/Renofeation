#!/bin/bash
export PYTHONPATH=../..:$PYTHONPATH

iter=5000
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

for i in 0 1
do



    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    NAME=resnet18_${DATASET_ABBR}_do_\
total${total_ratio}_init${init_ratio}_per${per_ratio}_int${interval}_trainall_\
lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${id}
    # DIR=results/datasetgrad/global_datasetgrad_divmag/lrschedule/trial${trial_iter}
    DIR=results/plot/feat_dist

    FINETUNE_DIR=results/baseline/finetune/resnet18_${DATASET_ABBR}_lr5e-3_iter30000_feat0_wd1e-4_mmt0_1
    RETRAIN_DIR=results/baseline/retrain/resnet18_${DATASET_ABBR}_reinit_lr1e-2_iter90000_feat0_wd5e-3_mmt0.9_1
    MY_DIR=results/datasetgrad/global_datasetgrad_divmag/lrschedule/trial3000/\
resnet18_${DATASET_ABBR}_do_total0.05_init0.05_per0_int10000_trainall_lr5e-3_iter30000_feat0_wd1e-4_mmt0_1
    RENO_DIR=results/baseline/renofeation1/resnet18_${DATASET_ABBR}_reinit_lr1e-2_iter90000_feat5e0_wd5e-3_mmt0.9_0
    # RENO_DIR=results/baseline/renofeation2/resnet18_${DATASET_ABBR}_reinit_constlr_lr5e-3_do1e-1_iter30000_feat5e0_wd5e-3_mmt0.9_1

    CUDA_VISIBLE_DEVICES=0 \
    python -u fineprune/plot/feat_dist.py \
    --iterations ${iter} \
    --datapath data/${DATASET}/ \
    --dataset ${DATASET_NAME} \
    --name ${NAME} \
    --batch_size 16 \
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
    --method weight \
    --train_all \
    --dropout 1e-1 \
    --finetune_ckpt $FINETUNE_DIR/ckpt.pth \
    --retrain_ckpt $RETRAIN_DIR/ckpt.pth \
    --my_ckpt $MY_DIR/ckpt.pth \
    --renofeation_ckpt $RENO_DIR/ckpt.pth \
    &
    
    # exit

    # WORKDIR=${DIR}/${NAME}
    # python -u eval_robustness.py \
    # --datapath data/${DATASET}/ \
    # --dataset ${DATASET_NAME} \
    # --network resnet18 \
    # --checkpoint ${WORKDIR}/ckpt.pth \
    # --batch_size 8 \
    # > ${WORKDIR}/eval.log


done