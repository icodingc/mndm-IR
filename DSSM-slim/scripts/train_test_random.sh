#!/bin/bash
TRAIN_DIR=./log/inshop_reproduce_log/
DATASET_DIR=${HOME}/workshops/data/In-shop-IR/tfrecord_triplet
export CUDA_VISIBLE_DEVICES=0
python train_custom_queue.py \
  --dssm_model=dssm \
  --train_dir=${TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16_src \
  --mode_name=float128 \
  --preprocessing_name=vgg_16 \
  --num_epochs_per_decay=100 \
  --batch_size=10 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --learning_rate=0.0003 \
  --learning_rate_decay_factor=0.1 \
  --weight_decay=0.00004 \
