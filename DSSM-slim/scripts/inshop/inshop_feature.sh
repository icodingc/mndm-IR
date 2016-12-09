TRAIN_DIR=${1}
MODE=${2}
DATASET_DIR=${HOME}/data/In-shop-IR/tfrecord_train
export CUDA_VISIBLE_DEVICES=0
python extract_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=/tmp/ \
  --dataset_name=fashion_inshop \
  --dataset_split_name=${MODE} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16_src \
  --mode_name=float128 \
  --batch_size=32 \
  --store_dir=${HOME}/workshops/DSSM-slim/test/inshop \
  --store_prefix=${MODE} \
  --layer1=avgp5 \
