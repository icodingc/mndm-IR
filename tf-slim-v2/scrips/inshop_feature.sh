TRAIN_DIR=${1}
MODE=${2}
DATASET_DIR=${HOME}/data/inshop/tfrecord_train
alias python1=/home/zhangxuesen/anaconda/bin/python
export CUDA_VISIBLE_DEVICES=3
python1 extract_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/ \
  --dataset_name=fashion_inshop \
  --dataset_split_name=${MODE} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16_src \
  --mode_name=float128 \
  --batch_size=64 \
  --store_dir=./test/inshop \
  --store_prefix=${MODE} \
  --layer1=avgp5 \
