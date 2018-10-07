# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RetinaNet (via ResNet) model definition.

Defines the RetinaNet model and loss functions from this paper:

https://arxiv.org/pdf/1708.02002

Uses the ResNet model as a basis.
"""

import numpy as np
import tensorflow as tf
from ..inception import inception_v4

slim = tf.contrib.slim

_WEIGHT_DECAY = 1e-4
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4
_RESNET_MAX_LEVEL = 5


def batch_norm_relu(inputs,
                    is_training_bn,
                    relu=True,
                    init_zero=False,
                    data_format='channels_last',
                    name=None):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training_bn,
      fused=True,
      gamma_initializer=gamma_initializer,
      name=name)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


## RetinaNet specific layers
def class_net(images, num_classes, num_anchors=6, is_training_bn=False):
  """Class prediction network for RetinaNet."""
  for i in range(4):
    images = tf.layers.conv2d(
        images,
        256,
        kernel_size=(3, 3),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        activation=None,
        padding='same',
        name='class-%d' % i)
    # The convolution layers in the class net are shared among all levels, but
    # each level has its batch normlization to capture the statistical
    # difference among different levels.
    images = batch_norm_relu(images, is_training_bn, relu=True, init_zero=False,
                             name='class-%d-bn' % i)

  classes = tf.layers.conv2d(
      images,
      num_classes * num_anchors,
      kernel_size=(3, 3),
      bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding='same',
      name='class-predict')

  return classes


def box_net(images, num_anchors=6, is_training_bn=False):
  """Box regression network for RetinaNet."""
  for i in range(4):
    images = tf.layers.conv2d(
        images,
        256,
        kernel_size=(3, 3),
        activation=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        padding='same',
        name='box-%d' % i)
    # The convolution layers in the box net are shared among all levels, but
    # each level has its batch normlization to capture the statistical
    # difference among different levels.
    images = batch_norm_relu(images, is_training_bn, relu=True, init_zero=False,
                             name='box-%d-bn-%d' % i)

  boxes = tf.layers.conv2d(
      images,
      4 * num_anchors,
      kernel_size=(3, 3),
      bias_initializer=tf.zeros_initializer(),
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding='same',
      name='box-predict')

  return boxes


def inception_net(inputs, is_training=True,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 scope='InceptionV4'):
    end_points = {}
    with tf.variable_scope(scope, 'InceptionV4', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = inception_v4.inception_v4_base(inputs, scope=scope)
    net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
    return net, end_points


def retinanet_no_fpn(features,
              min_level=3,
              max_level=7,
              num_classes=90,
              num_anchors=6,
              resnet_depth=50,
              use_nearest_upsampling=True,
              is_training_bn=False):
  """RetinaNet classification and regression model."""
  # create feature pyramid networks
  net, end_points = inception_net(features, is_training_bn)
  # add class net and box net in RetinaNet. The class net and the box net are
  # shared among all the levels.
  with tf.variable_scope('retinanet'):
    class_outputs = {}
    box_outputs = {}
    with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
        class_outputs = class_net(net, num_classes,
                                         num_anchors, is_training_bn)
    with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
        box_outputs = box_net(net,
                                     num_anchors, is_training_bn)

  return class_outputs, box_outputs

