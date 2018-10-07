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
from absl import flags

_WEIGHT_DECAY = 1e-4
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4
_RESNET_MAX_LEVEL = 5

FLAGS = flags.FLAGS
flags.DEFINE_bool("use_bottleneck", False, "Use bottleneck convolution layers")


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


def conv(image, filters, strides=1, kernel_size=3):
    """Convolution with default options from the densenet paper."""
    # Use initialization from https://arxiv.org/pdf/1502.01852.pdf

    return tf.layers.conv2d(
        inputs=image,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=tf.identity,
        use_bias=False,
        padding="same",
        kernel_initializer=tf.variance_scaling_initializer(),
    )


def dense_block(image, filters, is_training):
    """Standard BN+Relu+conv block for DenseNet."""
    image = tf.layers.batch_normalization(
        inputs=image,
        axis=-1,
        training=is_training,
        fused=True,
        center=True,
        scale=True,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
    )

    if FLAGS.use_bottleneck:
        # Add bottleneck layer to optimize computation and reduce HBM space
        image = tf.nn.relu(image)
        image = conv(image, 4 * filters, strides=1, kernel_size=1)
        image = tf.layers.batch_normalization(
            inputs=image,
            axis=-1,
            training=is_training,
            fused=True,
            center=True,
            scale=True,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
        )

    image = tf.nn.relu(image)
    return conv(image, filters)


def transition_layer(image, filters, is_training):
    """Construct the transition layer with specified growth rate."""

    image = tf.layers.batch_normalization(
        inputs=image,
        axis=-1,
        training=is_training,
        fused=True,
        center=True,
        scale=True,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
    )
    image = tf.nn.relu(image)
    conv_img = conv(image, filters=filters, kernel_size=1)
    return tf.layers.average_pooling2d(
        conv_img, pool_size=2, strides=2, padding="same")


def _int_shape(layer):
    return layer.get_shape().as_list()


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


def create_block(input, n, depth, is_training_bn=False):
    k = 32
    for j in range(depth):
        with tf.variable_scope("denseblock-%d-%d" % (n, j)):
            output = dense_block(input, k, is_training_bn)
            input = tf.concat([input, output], axis=3)
    return input


def densenet():
    k = 32

    def model(inputs, is_training_bn=False):
        num_channels = 2 * k
        v = conv(inputs, filters=2 * k, strides=2, kernel_size=7)
        v = tf.layers.batch_normalization(
            inputs=v,
            axis=-1,
            training=is_training_bn,
            fused=True,
            center=True,
            scale=True,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
        )
        v = tf.nn.relu(v)
        v = tf.layers.max_pooling2d(v, pool_size=3, strides=2, padding="same")

        c2 = create_block(v, 0, 6, is_training_bn)
        num_channels += k
        num_channels /= 2
        c2 = transition_layer(c2, num_channels, is_training_bn)

        c3 = create_block(c2, 1, 12, is_training_bn)
        num_channels += k
        num_channels /= 2
        c3 = transition_layer(c3, num_channels, is_training_bn)

        c4 = create_block(c3, 2, 24, is_training_bn)
        num_channels += k
        num_channels /= 2
        c4 = transition_layer(c4, num_channels, is_training_bn)

        c5 = create_block(c4, 3, 16, is_training_bn)

        return c2, c3, c4, c5

    return model


## RetinaNet specific layers
def class_net(images, level, num_classes, num_anchors=6, is_training_bn=False):
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
                                 name='class-%d-bn-%d' % (i, level))

    classes = tf.layers.conv2d(
        images,
        num_classes * num_anchors,
        kernel_size=(3, 3),
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        padding='same',
        name='class-predict')

    return classes


def box_net(images, level, num_anchors=6, is_training_bn=False):
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
                                 name='box-%d-bn-%d' % (i, level))

    boxes = tf.layers.conv2d(
        images,
        4 * num_anchors,
        kernel_size=(3, 3),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        padding='same',
        name='box-predict')

    return boxes


def densenet_fpn(features,
                 min_level=3,
                 max_level=7,
                 is_training_bn=False,
                 use_nearest_upsampling=True):
    with tf.variable_scope('densenet121'):
        densenet_fn = densenet()
        u2, u3, u4, u5 = densenet_fn(features, is_training_bn)

    feats_bottom_up = {
        2: u2,
        3: u3,
        4: u4,
        5: u5,
    }

    with tf.variable_scope("densenet_fpn"):
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
              max_level=7,
              num_classes=90,
              num_anchors=6,
              use_nearest_upsampling=True,
              is_training_bn=False):
    """RetinaNet classification and regression model."""
    # create feature pyramid networks
    feats = densenet_fpn(features, min_level, max_level,
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


def remove_variables(variables, resnet_depth=50):
    """Removes low-level variables from the input.

    Removing low-level parameters (e.g., initial convolution layer) from training
    usually leads to higher training speed and slightly better testing accuracy.
    The intuition is that the low-level architecture (e.g., ResNet-50) is able to
    capture low-level features such as edges; therefore, it does not need to be
    fine-tuned for the detection task.

    Args:
      variables: all the variables in training
      resnet_depth: the depth of ResNet model

    Returns:
      var_list: a list containing variables for training

    """
    var_list = [v for v in variables
                if v.name.find('densenet%s/conv2d/' % resnet_depth) == -1]
    return var_list


def segmentation_class_net(images,
                           level,
                           num_channels=256,
                           is_training_bn=False):
    """Segmentation Feature Extraction Module.

    Args:
      images: A tensor of size [batch, height_in, width_in, channels_in].
      level: The level of features at FPN output_size /= 2^level.
      num_channels: The number of channels in convolution layers
      is_training_bn: Whether batch_norm layers are in training mode.
    Returns:
      images: A feature tensor of size [batch, output_size, output_size,
        channel_number]
    """

    for i in range(3):
        images = tf.layers.conv2d(
            images,
            num_channels,
            kernel_size=(3, 3),
            bias_initializer=tf.zeros_initializer(),
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            activation=None,
            padding='same',
            name='class-%d' % i)
        images = batch_norm_relu(images, is_training_bn, relu=True, init_zero=False,
                                 name='class-%d-bn-%d' % (i, level))
    images = tf.layers.conv2d(
        images,
        num_channels,
        kernel_size=(3, 3),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        activation=None,
        padding='same',
        name='class-final')
    return images


def retinanet_segmentation(features,
                           min_level=3,
                           max_level=5,
                           num_classes=21,
                           resnet_depth=50,
                           use_nearest_upsampling=False,
                           is_training_bn=False):
    """RetinaNet extension for semantic segmentation.

    Args:
      features: A tensor of size [batch, height_in, width_in, channels].
      min_level: The minimum output feature pyramid level. This input defines the
        smallest nominal feature stride = 2^min_level.
      max_level: The maximum output feature pyramid level. This input defines the
        largest nominal feature stride = 2^max_level.
      num_classes: Number of object classes.
      resnet_depth: The depth of ResNet backbone model.
      use_nearest_upsampling: Whether use nearest upsampling for FPN network.
        Alternatively, use bilinear upsampling.
      is_training_bn: Whether batch_norm layers are in training mode.
    Returns:
      A tensor of size [batch, height_l, width_l, num_classes]
        representing pixel-wise predictions before Softmax function.
    """
    feats = densenet_fpn(features, min_level, max_level, resnet_depth,
                         is_training_bn, use_nearest_upsampling)

    with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
        for level in range(min_level, max_level + 1):
            feats[level] = segmentation_class_net(
                feats[level], level, is_training_bn=is_training_bn)
            if level == min_level:
                fused_feature = feats[level]
            else:
                if use_nearest_upsampling:
                    scale = level / min_level
                    feats[level] = nearest_upsampling(feats[level], scale)
                else:
                    feats[level] = resize_bilinear(
                        feats[level], tf.shape(feats[min_level])[1:3], feats[level].dtype)
                fused_feature += feats[level]
    fused_feature = batch_norm_relu(
        fused_feature, is_training_bn, relu=True, init_zero=False)
    classes = tf.layers.conv2d(
        fused_feature,
        num_classes,
        kernel_size=(3, 3),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        padding='same',
        name='class-predict')

    return classes
