from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

slim = tf.contrib.slim
def inference(inputs):
  with slim.arg_scope(vgg_arg_scope()):
    return vgg_16(inputs)


def vgg_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc
def vgg_16(inputs,
           is_training=False,
           scope='vgg_16'):
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1',trainable=False)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2',trainable=False)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3',trainable=False)
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4',trainable=False)
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      maxpool4 = net
      batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS}
      with slim.arg_scope([slim.conv2d],
                       normalizer_fn=slim.batch_norm,
                       normalizer_params=batch_norm_params,
                       outputs_collections=end_points_collection):
        with slim.arg_scope([slim.batch_norm],
                       is_training=is_training):
          net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
          net = slim.max_pool2d(net, [2, 2], scope='pool5')
          end_points = slim.utils.convert_collection_to_dict(end_points_collection)
          net = tf.squeeze(slim.avg_pool2d(net,[7,7],scope='avgp5'))
          return net

