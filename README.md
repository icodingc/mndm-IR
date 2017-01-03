# r0.12
- 用于保存以前的东西，固定checkpoint一下 
- 增加一些帮助代码，以及ipynb

# tf-slim-v2
- cars 数据集
- 增加batch_dssm 的训练方式，更有效的利用整个batch 的样本

# fixed bug
- batch_norm 阶段设置问题,抽feature时候is_training=False
- fine-tune 问题,把tf.trainable_variable()改为tf.all_variable()
