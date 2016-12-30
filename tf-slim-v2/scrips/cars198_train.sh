#!/bin/bash
TRAIN_DIR=./log/cars_log/cars_random_ftall_batch_loss
export CUDA_VISIBLE_DEVICES=3
alias python1=/home/zhangxuesen/anaconda/bin/python
python1 train_custom_queue_batch.py \
  --dssm_model=dssm \
  --train_dir=${TRAIN_DIR} \
  --model_name=vgg_16_bn \
  --mode_name=float128 \
  --preprocessing_name=vgg_16 \
  --num_epochs_per_decay=100 \
  --batch_size=10 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --learning_rate=0.01 \
  --learning_rate_decay_factor=0.1 \
  --weight_decay=0.00004 \
