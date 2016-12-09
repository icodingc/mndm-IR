## deepbay上的Tensorflow是0.9需要升级,使用此代码需要0.10以上。
此项目根据[Tf-slim](https://github.com/tensorflow/models/tree/master/slim)改写只有triplet_loss的训练而不包括分类，如有需要自己添加。
1. 首先生成triplet_loss 需要的三元组，写成`hard_and_random_triplet.lst`实例中的形式。
```
anchor.jpg\tpositive.jpg\t\neg1.jpg\t\neg2.jpg\n # 我train的时候一个anchor对应5个负例
```
2. 生成Tf训练的格式Tfrecord[代码](./build_fashion_triplet.py),把代码中293行改成第一步生成的.lst文件即可，程序运行完会生成80个tfrecord.
3. 更改[数据集文件](./datasets/fashion_inshop_triplet.py)中的18行的数字，改为你第一步生成的样本行数。
4. 训练[脚本](./train_tripletfloat128_random.sh)
```
#!/bin/bash
TRAIN_DIR=./log/inshop_log/          # 保存模型DIR
DATASET_DIR=${HOME}/workshops/data/In-shop-IR/tfrecord_triplet  # 第二步生成tfrecord 文件夹
export CUDA_VISIBLE_DEVICES=1                                   # 使用第几块GPU
python train_image_classifier.py \
  --dssm=True \
  --dssm_model=svm \
  --alpha=0.5 \                                                 #triplet_loss 中的marin
  --train_dir=${TRAIN_DIR} \
  --dataset_name=fashion_inshop_triplet \
  --dataset_split_name=random \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16_src \                                     #使用vgg网络
  --mode_name=float128 \
  --preprocessing_name=vgg_16 \
  --max_number_of_steps=100000 \
  --num_epochs_per_decay=40 \
  --batch_size=20 \
  --save_interval_secs=4600 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --learning_rate=0.01 \
  --learning_rate_decay_factor=0.1 \
  --weight_decay=0.00004 \
  --ignore_missing_vars=True \
  --checkpoint_path=./pre_train/                                #restore checkpoint我使用的是imagenet上train好的VGGmodel
```
具体都什么意思可以看[代码](./train_image_classifier.py)

