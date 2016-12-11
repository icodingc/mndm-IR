# dssm-slim
# fixed bug
- batch_norm 阶段设置问题,抽feature时候is_training=False
- fine-tune 问题,把tf.trainable_variable()改为tf.all_variable()
