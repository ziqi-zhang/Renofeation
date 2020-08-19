DATASETS=(MIT_67 CUB_200_2011 Flower_102 stanford_40 stanford_dog)
DATASET_NAMES=(MIT67Data CUB200Data Flower102Data Stanford40Data SDog120Data)
DATASET_ABBRS=(mit67 cub200 flower102 stanford40 sdog120)

for i in 0 
do
    DATASET=${DATASETS[i]}
    DATASET_NAME=${DATASET_NAMES[i]}
    DATASET_ABBR=${DATASET_ABBRS[i]}

    # retrain
    python -u eval_robustness.py \
    --datapath /data/MIT_67/ \
    --dataset MIT67Data \
    --network resnet18 \
    --checkpoint ckpt/resnet18_mit67_reinit_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.pth \
    > eval/resnet18_mit67_reinit_lr${lr}_iter${iter}_feat${lmda}_wd${wd}_mmt${mmt}_${i}.log



done