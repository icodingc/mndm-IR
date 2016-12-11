"""
@@_triplet_multi_loss
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.framework import deprecated

slim = tf.contrib.slim

#################################
#### DSSM
#################################
def _dssm_loss(anchor,positive,negatives,gamma,scope=None):
  with tf.name_scope(scope,'DSSM',[anchor,positive,negatives]):
    rst = [_cosine_distance(anchor,n) for n in [positive]+negatives]
    gamma = tf.convert_to_tensor(gamma,
                                  dtype=anchor.dtype.base_dtype,
                                  name='gamma_smooth')
    # batch*(p+n) 50*3 for dssm
    logits = tf.concat(1,rst)
    print('logits.shape',logits)
    p = tf.nn.softmax(gamma*logits)
    rst_loss = tf.reduce_mean(-tf.log(tf.squeeze(tf.slice(p,[0,0],[-1,1])),name='Dssmloss'))
    slim.losses.add_loss(rst_loss)
    return rst_loss
##################
### cos distance
##################
def _cosine_distance(a,b):
  a.get_shape().assert_is_compatible_with(b.get_shape())
  return tf.expand_dims(tf.reduce_sum(tf.mul(a,b),1),1)
##################
###  L2 distance
##################
def _euclidean_distance(a,b):
  a.get_shape().assert_is_compatible_with(b.get_shape())
  return tf.expand_dims(tf.reduce_sum(tf.square(tf.sub(a,b)),1),1)