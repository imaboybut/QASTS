#!/bin/bash
set -e

source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

model=ast
dataset=esc50
imagenetpretrain=True
audiosetpretrain=True
bal=none
lr=1e-5
freqm=24
timem=96
mixup=0
epoch=25
batch_size=48
fstride=10
tstride=10

dataset_mean=-6.6268077
dataset_std=5.358466
audio_length=512
noise=False

metrics=acc
loss=CE
warmup=False
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85

for((fold=1;fold<=5;fold++));
do
  exp_dir=./exp/fp_fold${fold}_baseline
  tr_data=./data/datafiles/esc_train_data_${fold}.json
  te_data=./data/datafiles/esc_eval_data_${fold}.json

  rm -rf "$exp_dir"

  CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir "$exp_dir" \
  --label-csv ./data/esc_class_labels_indices.csv --n_class 50 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain \
  --metrics ${metrics} --loss ${loss} --warmup ${warmup} \
  --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise}
done
