#!/bin/bash

iter=10000
id=1
splmda=0
lmda=0
layer=1234
mmt=0

DATASETS=(MIT_67 stanford_dog GTSRB CUB_200_2011 Flower_102 stanford_40 stanford_dog GTSRB)
DATASET_NAMES=(MIT67Data SDog120Data GTSRBData CUB200Data Flower102Data Stanford40Data SDog120Data GTSRBData)
DATASET_ABBRS=(mit67 sdog120 gtsrb cub200 flower102 stanford40 sdog120 gtsrb)
LEARNING_RATE=5e-3
WEIGHT_DECAY=1e-4
PORTION=(0.2 0.5 0.7 0.9)
RATIO=(0.0 0.5 0.7 0.9)

for i in 0 
do
    for j in 0 1 3
    do
    for k in 0 1
    do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}
    lr=${LEARNING_RATE}
    wd=${WEIGHT_DECAY}
    portion=${PORTION[j]}
    ratio=${RATIO[k]}
    #NAME=fixed_${DATASET_ABBR}_${ratio}

    NAME=one_${DATASET_ABBR}_${ratio}_${portion}
    newDIR=results/qianyi_0725/

    CUDA_VISIBLE_DEVICES=$k \
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
    --const_lr \
    --argportion ${portion} \
    --backdoor_update_ratio ${ratio} \
    --teacher_method backdoor_finetune \

    &
    
    done
    done
done

