# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def vgg_16_src(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           mode='float128',
           scope='vgg_16'):
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
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
          #TODO conv5
          net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
          net = slim.max_pool2d(net, [2, 2], scope='pool5')
          end_points = slim.utils.convert_collection_to_dict(end_points_collection)
          net = slim.avg_pool2d(net,[7,7],scope='avgp5')
          end_points['avgp5'] = net
          net = slim.conv2d(net,128,[1,1],activation_fn=tf.tanh,scope='embedding')
          net = tf.squeeze(net)
          end_points['embedding'] = net
          return net, end_points
vgg_16_src.default_image_size = 224
def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           mode='float128',
           scope='vgg_16'):
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
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
      #TODO conv5
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      net = slim.avg_pool2d(net,[7,7],scope='avgp5')
      end_points['avgp5'] = net
      return net, end_points
vgg_16.default_image_size = 224
def vgg_16_bn(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           mode='float128',
           scope='vgg_16'):
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
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
	  net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
	  net = slim.max_pool2d(net, [2, 2], scope='pool1')
	  net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
	  net = slim.max_pool2d(net, [2, 2], scope='pool2')
	  net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
	  net = slim.max_pool2d(net, [2, 2], scope='pool3')
	  net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
	  net = slim.max_pool2d(net, [2, 2], scope='pool4')
          #TODO conv5
          net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
          net = slim.max_pool2d(net, [2, 2], scope='pool5')
          end_points = slim.utils.convert_collection_to_dict(end_points_collection)
          net = slim.avg_pool2d(net,[7,7],scope='avgp5')
          end_points['avgp5'] = net
          net = slim.conv2d(net,128,[1,1],activation_fn=tf.tanh,scope='embedding')
          net = tf.squeeze(net)
          end_points['embedding'] = net
          return net, end_points
vgg_16_bn.default_image_size = 224
