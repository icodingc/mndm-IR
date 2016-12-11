#!/bin/bash
TRAIN_DIR=./log/inshop_dssm_random2/
DATASET_DIR=${HOME}/data/tfrecord_triplet
export CUDA_VISIBLE_DEVICES=3
alias python1=/home/zhangxuesen/anaconda/bin/python
python1 train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=fashion_inshop_triplet \
  --dataset_split_name=random \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16_src \
  --mode_name=float128 \
  --preprocessing_name=vgg_16 \
  --max_number_of_steps=100000 \
  --num_epochs_per_decay=100 \
  --batch_size=10 \
  --save_interval_secs=3600 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --learning_rate=0.01 \
  --learning_rate_decay_factor=0.1 \
  --weight_decay=0.00004 \
  --ignore_missing_vars=True \
  --checkpoint_path=./log/inshop_dssm_random/model.ckpt-13025
