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


def block_inception_a(inputs, scope=None, reuse=None):
    """Builds Inception-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInceptionA', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 96, [1, 1], scope='Conv2d_0b_1x1')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])


def block_reduction_a(inputs, scope=None, reuse=None):
    """Builds Reduction-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockReductionA', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 384, [3, 3], stride=2, padding='VALID',
                                       scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                branch_1 = slim.conv2d(branch_1, 256, [3, 3], stride=2,
                                       padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                           scope='MaxPool_1a_3x3')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])


def block_inception_b(inputs, scope=None, reuse=None):
    """Builds Inception-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInceptionB', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 224, [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 256, [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, 224, [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, 224, [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, 256, [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])


def block_reduction_b(inputs, scope=None, reuse=None):
    """Builds Reduction-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockReductionB', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_0 = slim.conv2d(branch_0, 192, [3, 3], stride=2,
                                       padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 256, [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 320, [7, 1], scope='Conv2d_0c_7x1')
                branch_1 = slim.conv2d(branch_1, 320, [3, 3], stride=2,
                                       padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                           scope='MaxPool_1a_3x3')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])


def block_inception_c(inputs, scope=None, reuse=None):
    """Builds Inception-C block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInceptionC', [inputs], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = tf.concat(axis=3, values=[
                    slim.conv2d(branch_1, 256, [1, 3], scope='Conv2d_0b_1x3'),
                    slim.conv2d(branch_1, 256, [3, 1], scope='Conv2d_0c_3x1')])
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 448, [3, 1], scope='Conv2d_0b_3x1')
                branch_2 = slim.conv2d(branch_2, 512, [1, 3], scope='Conv2d_0c_1x3')
                branch_2 = tf.concat(axis=3, values=[
                    slim.conv2d(branch_2, 256, [1, 3], scope='Conv2d_0d_1x3'),
                    slim.conv2d(branch_2, 256, [3, 1], scope='Conv2d_0e_3x1')])
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 256, [1, 1], scope='Conv2d_0b_1x1')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])


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


def nearest_upsampling(data, scale):
    """Nearest neighbor upsampling implementation.

    Args:
      data: A float32 tensor of size [batch, height_in, width_in, channels].
      scale: An integer multiple to scale resolution of input data.
    Returns:
      data_up: A float32 tensor of size
        [batch, height_in*scale, width_in*scale, channels].
    """
    with tf.name_scope('nearest_upsampling'):
        bs, h, w, c = data.get_shape().as_list()
        bs = -1 if bs is None else bs
        # Use reshape to quickly upsample the input.  The nearest pixel is selected
        # implicitly via broadcasting.
        data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
            [1, 1, scale, 1, scale, 1], dtype=data.dtype)
        return tf.reshape(data, [bs, h * scale, w * scale, c])


# TODO(b/111271774): Removes this wrapper once b/111271774 is resolved.
def resize_bilinear(images, size, output_type):
    """Returns resized images as output_type.

    Args:
      images: A tensor of size [batch, height_in, width_in, channels].
      size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size
        for the images.
      output_type: The destination type.
    Returns:
      A tensor of size [batch, height_out, width_out, channels] as a dtype of
        output_type.
    """
    images = tf.image.resize_bilinear(images, size, align_corners=True)
    return tf.cast(images, output_type)


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


def inception_builder():
    scope = "InceptionV4"

    def model(inputs, is_training_bn=False):
        with tf.variable_scope(scope, 'InceptionV4', [inputs]):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # 299 x 299 x 3
                net = slim.conv2d(inputs, 32, [3, 3], stride=2,
                                  padding='VALID', scope='Conv2d_1a_3x3')
                # 149 x 149 x 32
                net = slim.conv2d(net, 32, [3, 3], padding='VALID',
                                  scope='Conv2d_2a_3x3')
                # 147 x 147 x 32
                net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_2b_3x3')
                # 147 x 147 x 64
                with tf.variable_scope('Mixed_3a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                   scope='MaxPool_0a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 96, [3, 3], stride=2, padding='VALID',
                                               scope='Conv2d_0a_3x3')
                    c2 = tf.concat(axis=3, values=[branch_0, branch_1])

                # 73 x 73 x 160
                with tf.variable_scope('Mixed_4a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(c2, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='VALID',
                                               scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(c2, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], padding='VALID',
                                               scope='Conv2d_1a_3x3')
                    c3 = tf.concat(axis=3, values=[branch_0, branch_1])

                # 71 x 71 x 192
                with tf.variable_scope('Mixed_5a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(c3, 192, [3, 3], stride=2, padding='VALID',
                                               scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.max_pool2d(c3, [3, 3], stride=2, padding='VALID',
                                                   scope='MaxPool_1a_3x3')
                    c3 = tf.concat(axis=3, values=[branch_0, branch_1])

                # 35 x 35 x 384
                # 4 x Inception-A blocks
                c4 = c3
                for idx in range(4):
                    block_scope = 'Mixed_5' + chr(ord('b') + idx)
                    c4 = block_inception_a(c4, block_scope)
                # 35 x 35 x 384
                # Reduction-A block
                c4 = block_reduction_a(c4, 'Mixed_6a')

                c5 = c4
                # 17 x 17 x 1024
                # 7 x Inception-B blocks
                for idx in range(7):
                    block_scope = 'Mixed_6' + chr(ord('b') + idx)
                    c5 = block_inception_b(c5, block_scope)
                # 17 x 17 x 1024
                # Reduction-B block
                c5 = block_reduction_b(c5, 'Mixed_7a')

                # 8 x 8 x 1536
                # 3 x Inception-C blocks
                for idx in range(3):
                    block_scope = 'Mixed_7' + chr(ord('b') + idx)
                    c5 = block_inception_c(c5, block_scope)
        return c2, c3, c4, c5
    return model


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


def inception_fpn(features,
                 min_level=3,
                 max_level=7,
                 is_training_bn=False,
                 use_nearest_upsampling=True):
    with tf.variable_scope("inceptionv4"):
        inception_fn = inception_builder()
        u2, u3, u4, u5 = inception_fn(features, is_training_bn)

    feats_bottom_up = {
        2: u2,
        3: u3,
        4: u4,
        5: u5,
    }

    with tf.variable_scope("inception_fpn"):
        feats_lateral = {}
        for level in range(min_level, _RESNET_MAX_LEVEL + 1):
            feats_lateral[level] = tf.layers.conv2d(
                feats_bottom_up[level],
                filters=256,
                kernel_size=(1, 1),
                padding='same',
                name='1%d' % level)
        feats = {_RESNET_MAX_LEVEL: feats_lateral[_RESNET_MAX_LEVEL]}
        for level in range(_RESNET_MAX_LEVEL - 1, min_level - 1, -1):
            if use_nearest_upsampling:
                feats[level] = nearest_upsampling(feats[level + 1], 2) + feats_lateral[level]
            else:
                feats[level] = resize_bilinear(
                    feats[level + 1], tf.shape(feats_lateral[level])[1:3],
                    feats[level + 1].dtype) + feats_lateral[level]

        for level in range(min_level, _RESNET_MAX_LEVEL + 1):
            feats[level] = tf.layers.conv2d(
                feats[level],
                filters=256,
                strides=(1, 1),
                kernel_size=(3, 3),
                padding='same',
                name='post_hoc_d%d' % level)

        for level in range(_RESNET_MAX_LEVEL + 1, max_level + 1):
            feats_in = feats[level - 1]
            if level > _RESNET_MAX_LEVEL + 1:
                feats_in = tf.nn.relu(feats_in)
            feats[level] = tf.layers.conv2d(
                feats_in,
                filters=256,
                strides=(2, 2),
                kernel_size=(3, 3),
                padding='same',
                name='p%d' % level)
            # add batchnorm
        for level in range(min_level, max_level + 1):
            feats[level] = tf.layers.batch_normalization(
                inputs=feats[level],
                momentum=_BATCH_NORM_DECAY,
                epsilon=_BATCH_NORM_EPSILON,
                center=True,
                scale=True,
                training=is_training_bn,
                fused=True,
                name='p%d-bn' % level)
    return feats


def retinanet(features,
              min_level=3,
              max_level=9,
              num_classes=90,
              num_anchors=6,
              use_nearest_upsampling=True,
              is_training_bn=False):
    feats = inception_fpn(features, min_level, max_level,
                         is_training_bn, use_nearest_upsampling)
    # add class net and box net in RetinaNet. The class net and the box net are
    # shared among all the levels.
    with tf.variable_scope('retinanet'):
        class_outputs = {}
        box_outputs = {}
        with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
            for level in range(min_level, max_level + 1):
                class_outputs[level] = class_net(feats[level], level, num_classes,
                                                 num_anchors, is_training_bn)
        with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
            for level in range(min_level, max_level + 1):
                box_outputs[level] = box_net(feats[level], level,
                                             num_anchors, is_training_bn)

    return class_outputs, box_outputs