#!/bin/bash
set -e

source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

model=ast
dataset=fsdkaggle2018
imagenetpretrain=True
audiosetpretrain=True
bal=none
lr=1e-5
freqm=24
timem=96
mixup=0
epoch=25
batch_size=24
fstride=10
tstride=10

dataset_mean=-1.6813264
dataset_std=2.4526765
audio_length=1024
noise=True

metrics=acc
loss=CE
warmup=True
lrscheduler_start=10
lrscheduler_step=5
lrscheduler_decay=0.5

base_exp_dir=./exp/test-${dataset}-f$fstride-t$tstride-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}

tr_data=./data/datafiles/fsd_train_data.json
val_data=./data/datafiles/fsd_val_data.json
te_data=./data/datafiles/fsd_test_data.json

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${te_data} --exp-dir "$base_exp_dir" \
--label-csv ./data/fsd_class_labels_indices.csv --n_class 41 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise}
