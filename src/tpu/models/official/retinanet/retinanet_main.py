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
"""Training script for RetinaNet.
"""

import os

from absl import flags
import numpy as np
import tensorflow as tf

from . import dataloader
from . import retinanet_model

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
         'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
         'url.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

# Model specific paramenters
flags.DEFINE_string(
    'eval_master', default='',
    help='GRPC URL of the eval master. Set to an appropiate value when running '
         'on CPU/GPU')
flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than CPUs')
flags.DEFINE_bool(
    'use_xla', False,
    'Use XLA even if use_tpu is false.  If use_tpu is true, we always use XLA, '
    'and this flag has no effect.')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string('resnet_checkpoint', '',
                    'Location of the ResNet50 checkpoint to use for model '
                    'initialization.')
flags.DEFINE_string('densenet_checkpoint', '', 'Location of Densenet checkpoint')
flags.DEFINE_string('hparams', '',
                    'Comma separated k=v pairs of hyperparameters.')
flags.DEFINE_integer(
    'num_cores', default=8, help='Number of TPU cores for training')
flags.DEFINE_bool('use_spatial_partition', False, 'Use spatial partition.')
flags.DEFINE_integer(
    'num_cores_per_replica', default=8, help='Number of TPU cores per'
                                             'replica when using spatial partition.')
flags.DEFINE_multi_integer(
    'input_partition_dims', [1, 4, 2, 1],
    'A list that describes the partition dims for all the tensors.')
flags.DEFINE_integer('train_batch_size', 64, 'training batch size')
flags.DEFINE_integer('eval_batch_size', 1, 'evaluation batch size')
flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                                           'evaluation.')
flags.DEFINE_integer(
    'iterations_per_loop', 100, 'Number of iterations per TPU training loop')
flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string(
    'validation_file_pattern', None,
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string(
    'val_json_file',
    None,
    'COCO validation JSON containing golden bounding boxes.')
flags.DEFINE_integer('num_examples_per_epoch', 120000,
                     'Number of examples in one epoch')
flags.DEFINE_integer('num_epochs', 15, 'Number of epochs for training')
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                                                'training finishes.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.

    if FLAGS.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project)
        tpu_grpc_url = tpu_cluster_resolver.get_master()
        tf.Session.reset(tpu_grpc_url)
    else:
        tpu_cluster_resolver = None

    # Check data path
    if FLAGS.mode in ('train',
                      'train_and_eval') and FLAGS.training_file_pattern is None:
        raise RuntimeError('You must specify --training_file_pattern for training.')
    if FLAGS.mode in ('eval', 'train_and_eval'):
        if FLAGS.validation_file_pattern is None:
            raise RuntimeError('You must specify --validation_file_pattern '
                               'for evaluation.')
        if FLAGS.val_json_file is None:
            raise RuntimeError('You must specify --val_json_file for evaluation.')

    # Parse hparams
    hparams = retinanet_model.default_hparams()
    hparams.parse(FLAGS.hparams)

    # The following is for spatial partitioning. `features` has one tensor while
    # `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
    # partition is performed on `features` and all partitionable tensors of
    # `labels`, see the partition logic below.
    # In the TPUEstimator context, the meaning of `shard` and `replica` is the
    # same; follwing the API, here has mixed use of both.
    if FLAGS.use_spatial_partition:
        # Checks input_partition_dims agrees with num_cores_per_replica.
        if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
            raise RuntimeError('--num_cores_per_replica must be a product of array'
                               'elements in --input_partition_dims.')

        labels_partition_dims = {
            'mean_num_positives': None,
            'source_ids': None,
            'groundtruth_data': None,
            'image_scales': None,
        }
        # The Input Partition Logic: We partition only the partition-able tensors.
        # Spatial partition requires that the to-be-partitioned tensors must have a
        # dimension that is a multiple of `partition_dims`. Depending on the
        # `partition_dims` and the `image_size` and the `max_level` in hparams, some
        # high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
        # be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
        # size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
        # [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
        # case, the level-8 and level-9 target tensors are not partition-able, and
        # the highest partition-able level is 7.
        image_size = hparams.get('image_size')
        for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):

            def _can_partition(spatial_dim):
                partitionable_index = np.where(
                    spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
                return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

            spatial_dim = image_size // (2 ** level)
            if _can_partition(spatial_dim):
                labels_partition_dims[
                    'box_targets_%d' % level] = FLAGS.input_partition_dims
                labels_partition_dims[
                    'cls_targets_%d' % level] = FLAGS.input_partition_dims
            else:
                labels_partition_dims['box_targets_%d' % level] = None
                labels_partition_dims['cls_targets_%d' % level] = None

        num_cores_per_replica = FLAGS.num_cores_per_replica
        input_partition_dims = [
            FLAGS.input_partition_dims, labels_partition_dims]
        num_shards = FLAGS.num_cores // num_cores_per_replica
    else:
        num_cores_per_replica = None
        input_partition_dims = None
        num_shards = FLAGS.num_cores

    params = dict(
        hparams.values(),
        num_shards=num_shards,
        num_examples_per_epoch=FLAGS.num_examples_per_epoch,
        use_tpu=FLAGS.use_tpu,
        resnet_checkpoint=FLAGS.resnet_checkpoint,
        densenet_checkpoint=FLAGS.densenet_checkpoint,
        val_json_file=FLAGS.val_json_file,
        mode=FLAGS.mode,
    )
    config_proto = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    if FLAGS.use_xla and not FLAGS.use_tpu:
        config_proto.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1)

    tpu_config = tf.contrib.tpu.TPUConfig(
        FLAGS.iterations_per_loop,

        num_shards=num_shards,
        num_cores_per_replica=num_cores_per_replica,
        input_partition_dims=input_partition_dims,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    )

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        evaluation_master=FLAGS.eval_master,
        model_dir=FLAGS.model_dir,
        log_step_count_steps=FLAGS.iterations_per_loop,
        session_config=config_proto,
        tpu_config=tpu_config,
    )

    # TPU Estimator
    if FLAGS.mode == 'train':
        tf.logging.info(params)
        train_estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=retinanet_model.retinanet_model_fn,
            use_tpu=FLAGS.use_tpu,
            train_batch_size=FLAGS.train_batch_size,
            config=run_config,
            params=params)
        train_estimator.train(
            input_fn=dataloader.InputReader(FLAGS.training_file_pattern,
                                            is_training=True),
            max_steps=int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                          FLAGS.train_batch_size))

        if FLAGS.eval_after_training:
            # Run evaluation after training finishes.
            eval_params = dict(
                params,
                use_tpu=False,
                input_rand_hflip=False,
                resnet_checkpoint=None,
                densenet_checkpoint=None,
                is_training_bn=False,
                use_bfloat16=False,
            )
            eval_estimator = tf.contrib.tpu.TPUEstimator(
                model_fn=retinanet_model.retinanet_model_fn,
                use_tpu=False,
                train_batch_size=FLAGS.train_batch_size,
                eval_batch_size=FLAGS.eval_batch_size,
                config=run_config,
                params=eval_params)
            eval_results = eval_estimator.evaluate(
                input_fn=dataloader.InputReader(FLAGS.validation_file_pattern,
                                                is_training=False),
                steps=FLAGS.eval_samples // FLAGS.eval_batch_size)
            tf.logging.info('Eval results: %s' % eval_results)

    elif FLAGS.mode == 'eval':
        # Eval only runs on CPU or GPU host with batch_size = 1.
        # Override the default options: disable randomization in the input pipeline
        # and don't run on the TPU.
        # Also, disable use_bfloat16 for eval on CPU/GPU.
        eval_params = dict(
            params,
            use_tpu=False,
            input_rand_hflip=False,
            resnet_checkpoint=None,
            is_training_bn=False,
            use_bfloat16=False,
        )

        eval_estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=retinanet_model.retinanet_model_fn,
            use_tpu=False,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            config=run_config,
            params=eval_params)

        def terminate_eval():
            tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                            FLAGS.eval_timeout)
            return True

        # Run evaluation when there's a new checkpoint
        for ckpt in tf.contrib.training.checkpoints_iterator(
                FLAGS.model_dir,
                min_interval_secs=FLAGS.min_eval_interval,
                timeout=FLAGS.eval_timeout,
                timeout_fn=terminate_eval):

            tf.logging.info('Starting to evaluate.')
            try:
                eval_results = eval_estimator.evaluate(
                    input_fn=dataloader.InputReader(FLAGS.validation_file_pattern,
                                                    is_training=False),
                    steps=FLAGS.eval_samples // FLAGS.eval_batch_size)
                tf.logging.info('Eval results: %s' % eval_results)

                # Terminate eval job when final checkpoint is reached
                current_step = int(os.path.basename(ckpt).split('-')[1])
                total_step = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                                 FLAGS.train_batch_size)
                if current_step >= total_step:
                    tf.logging.info('Evaluation finished after training step %d' %
                                    current_step)
                    break

            except tf.errors.NotFoundError:
                # Since the coordinator is on a different job than the TPU worker,
                # sometimes the TPU worker does not finish initializing until long after
                # the CPU job tells it to start evaluating. In this case, the checkpoint
                # file could have been deleted already.
                tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint' %
                                ckpt)

    elif FLAGS.mode == 'train_and_eval':
        for cycle in range(FLAGS.num_epochs):
            tf.logging.info('Starting training cycle, epoch: %d.' % cycle)
            train_estimator = tf.contrib.tpu.TPUEstimator(
                model_fn=retinanet_model.retinanet_model_fn,
                use_tpu=FLAGS.use_tpu,
                train_batch_size=FLAGS.train_batch_size,
                config=run_config,
                params=params)
            train_estimator.train(
                input_fn=dataloader.InputReader(FLAGS.training_file_pattern,
                                                is_training=True),
                steps=int(FLAGS.num_examples_per_epoch / FLAGS.train_batch_size))

            tf.logging.info('Starting evaluation cycle, epoch: %d.' % cycle)
            # Run evaluation after every epoch.
            eval_params = dict(
                params,
                use_tpu=False,
                input_rand_hflip=False,
                resnet_checkpoint=None,
                densenet_checkpoint=None,
                is_training_bn=False,
            )

            eval_estimator = tf.contrib.tpu.TPUEstimator(
                model_fn=retinanet_model.retinanet_model_fn,
                use_tpu=False,
                train_batch_size=FLAGS.train_batch_size,
                eval_batch_size=FLAGS.eval_batch_size,
                config=run_config,
                params=eval_params)
            eval_results = eval_estimator.evaluate(
                input_fn=dataloader.InputReader(FLAGS.validation_file_pattern,
                                                is_training=False),
                steps=FLAGS.eval_samples // FLAGS.eval_batch_size)
            tf.logging.info('Evaluation results: %s' % eval_results)

    elif FLAGS.mode == 'predict':
        # Predict only runs on CPU or GPU host with batch_size = 1.
        # Override the default options: disable randomization in the input pipeline
        # and don't run on the TPU.
        # Also, disable use_bfloat16 for eval on CPU/GPU.
        predict_params = dict(
            params,
            use_tpu=False,
            input_rand_hflip=False,
            resnet_checkpoint=None,
            densenet_checkpoint=None,
            is_training_bn=False,
            use_bfloat16=False,
        )

        predict_estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=retinanet_model.retinanet_model_fn,
            use_tpu=False,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            config=run_config,
            params=predict_params)

        def terminate_eval():
            tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                            FLAGS.eval_timeout)
            return True

        # Run evaluation when there's a new checkpoint
        for ckpt in tf.contrib.training.checkpoints_iterator(
                FLAGS.model_dir,
                min_interval_secs=FLAGS.min_eval_interval,
                timeout=FLAGS.eval_timeout,
                timeout_fn=terminate_eval):

            filename = "results-" + os.path.basename(ckpt)
            f = open(filename, 'w')
            tf.logging.info('Starting to predict.')
            try:
                predict_results = predict_estimator.predict(
                    input_fn=dataloader.InputReader(FLAGS.validation_file_pattern,
                                                    is_training=False),
                    steps=FLAGS.eval_samples // FLAGS.eval_batch_size)
                for i, result in enumerate(predict_results):
                    tf.logging.info("Prediction result %d: %s" % (i, result))
                    f.write(result)
                    f.write("\n")

                # Terminate eval job when final checkpoint is reached
                current_step = int(os.path.basename(ckpt).split('-')[1])
                total_step = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
                                 FLAGS.train_batch_size)
                if current_step >= total_step:
                    tf.logging.info('Evaluation finished after training step %d' %
                                    current_step)
                    break

            except tf.errors.NotFoundError:
                # Since the coordinator is on a different job than the TPU worker,
                # sometimes the TPU worker does not finish initializing until long after
                # the CPU job tells it to start evaluating. In this case, the checkpoint
                # file could have been deleted already.
                tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint' %
                                ckpt)
            f.close()
    else:
        tf.logging.info('Mode not found.')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
