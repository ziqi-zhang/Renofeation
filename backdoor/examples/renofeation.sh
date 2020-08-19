#!/bin/bash

iter=30000
id=1
splmda=0
layer=1234
lr=1e-2
wd=5e-3
mmt=0.9
lmda=5e0

DATASETS=(MIT_67 stanford_dog GTSRB CUB_200_2011 Flower_102 stanford_40 stanford_dog GTSRB)
DATASET_NAMES=(MIT67Data SDog120Data GTSRBData CUB200Data Flower102Data Stanford40Data SDog120Data GTSRBData)
DATASET_ABBRS=(mit67 sdog120 gtsrb cub200 flower102 stanford40 sdog120 gtsrb)
LEARNING_RATE=5e-3
WEIGHT_DECAY=1e-4
PORTION=(0.2 0.5 0.7 0.9)
RATIO=(0.0 0.5 0.7 0.9)

for j in 0 
do
    for i in 0 2
    do

    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}
    lr=${LEARNING_RATE}
    wd=${WEIGHT_DECAY}
    portion=0.2
    ratio=${RATIO[j]}
    NAME=fixed_${DATASET_ABBR}_${ratio}
    #NAME=random_${DATASET_ABBR}_${ratio}
    newDIR=results/backdoor/renofeation1_30000
    # newDIR=results/test
    teacher_dir=results/backdoor/baseline/fixed_${DATASET_ABBR}_${ratio}

    CUDA_VISIBLE_DEVICES=$1 \
    python -u py_qianyi.py \
    --teacher_datapath ../data/${DATASET} \
    --teacher_dataset ${DATASET_NAME} \
    --student_datapath ../data/${DATASET} \
    --student_dataset ${DATASET_NAME} \
    --iterations ${iter} \
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
    --output_dir ${newDIR} \
    --argportion ${portion} \
    --backdoor_update_ratio ${ratio} \
    --teacher_method backdoor_finetune \
    --checkpoint $teacher_dir/teacher_ckpt.pth \
    --student_method finetune \
    --fixed_pic \
    --train_all \
    --prune_interval 100 \
    --weight_total_ratio 0.05 \
    --weight_ratio_per_prune 0 \
    --weight_init_prune_ratio 0.05 \
    --trial_iteration 3000 \
    --trial_lr ${lr} \
    --trial_momentum 0.9 \
    --trial_weight_decay 0 \
    --dropout 1e-1 \
    --reinit \
    &

    done
done

