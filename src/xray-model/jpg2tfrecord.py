import tensorflow as tf
import sys
from optparse import OptionParser
import os
from tensorflow.python.lib.io import file_io
import zipfile
import cv2
import shutil
import utils
import pandas as pd
import math

TRAINING_SHARDS = 1024
VALIDATION_SHARDS = 128
TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'

def prepare_folder(folder_path):
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)

    os.mkdir(folder_path)

def create_tf_example(image_buffer, label, width, height):
    image_format = b'png'
    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': utils.int64_feature(height),
        'image/width': utils.int64_feature(width),
        'image/encoded': utils.bytes_feature(image_buffer),
        'image/format': utils.bytes_feature(image_format),
        'image/object/class/label': utils.int64_feature(label)
    }))

def _process_image(filename):

  model_file = file_io.FileIO(filename, mode='rb')
  temp_model_location = './temp.png'
  temp_model_file = open(temp_model_location, 'wb')
  temp_model_file.write(model_file.read())
  temp_model_file.close()
  image_data = cv2.imread(temp_model_location)

  decoded = cv2.imencode('.png', image_data)[1].tostring()
  height, width, _ = image_data.shape
  return decoded, height, width

def _process_image_files_batch(output_file, filenames, synsets, labels):

  writer = tf.python_io.TFRecordWriter('tmp.record')

  for filename, synset in zip(filenames, synsets):
    image_buffer, height, width = _process_image(filename)
    label = labels[synset]
    example = create_tf_example(image_buffer, label, width, height)
    writer.write(example.SerializeToString())

  writer.close()

  with file_io.FileIO('tmp.record', mode='rb') as f:
      with file_io.FileIO(output_file, mode='wb+') as of:
          of.write(f.read())
          print(output_file, 'written')



def _process_dataset(filenames, synsets, labels, output_directory, prefix, num_shards):
  """Processes and saves list of images as TFRecords.
  Args:
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: map of string to integer; id for all synset labels
    output_directory: path where output files should be created
    prefix: string; prefix for each file
    num_shards: number of chucks to split the filenames into
  Returns:
    files: list of tf-record filepaths created from processing the dataset.
  """
  chunksize = int(math.ceil(len(filenames) / num_shards))

  files = []

  for shard in range(num_shards):
    chunk_files = filenames[shard * chunksize : (shard + 1) * chunksize]
    chunk_synsets = synsets[shard * chunksize : (shard + 1) * chunksize]
    output_file = os.path.join(
        output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
    _process_image_files_batch(output_file, chunk_files,
                               chunk_synsets, labels)


    tf.logging.info('Finished writing file: %s' % output_file)
    files.append(output_file)
  return files

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--gcsdir", dest="gcs_dir", help="GCS base dir")
    parser.add_option("-i", "--imgdir", dest="img_dir", help="PNNGs dir")
    parser.add_option("-t", "--tfrecorddir", dest="tfrecord_dir", help="tfrecords dir")

    (options, args) = parser.parse_args()

    if not options.gcs_dir:
        raise Exception('GCS dir not provided')

    if not options.img_dir:
        raise Exception('img_dir not provided')

    if not options.tfrecord_dir:
        raise Exception('tfrecord_dir not provided')


    with file_io.FileIO(os.path.join(options.gcs_dir, 'Data_Entry_2017.csv'), 'r') as f:
        dataentry = pd.read_csv(f)

    with file_io.FileIO(os.path.join(options.gcs_dir, 'ClassesLabel.csv'), 'r') as f:
        mapping_df = pd.read_csv(f)

    with file_io.FileIO(os.path.join(options.gcs_dir, 'split.csv'), 'r') as f:
        train_map_df = pd.read_csv(f)

    tf_records_path = os.path.join(options.gcs_dir, 'records')

    allowed_images = set([x['Image Index'] for _, x in dataentry.iterrows() if '|' not in x['Finding Labels']])

    allowed_images = set(list(os.walk(options.img_dir))[0][2])

    image_to_class = {x['Image Index']: x['Finding Labels'] for _, x in dataentry.iterrows() if '|' not in x['Finding Labels']}
    class_to_label = {x['Classes']: x['Label'] for _, x in mapping_df.iterrows()}

    training_names = [x['image'] for _, x in train_map_df.iterrows() if x['train'] == 1 and x['image'] in allowed_images]
    validation_names = [x['image'] for _, x in train_map_df.iterrows() if x['train'] == 0 and x['image'] in allowed_images]
    training_classes = [image_to_class[x] for x in training_names]
    validation_classes = [image_to_class[x] for x in validation_names]

    training_filenames = [os.path.join(options.img_dir, x) for x in training_names]
    validation_filenames = [os.path.join(options.img_dir, x) for x in validation_names]

    training = _process_dataset(training_filenames, training_classes, class_to_label, options.tfrecord_dir, TRAINING_DIRECTORY, TRAINING_SHARDS)
    validation = _process_dataset(validation_filenames, validation_classes, class_to_label, options.tfrecord_dir, VALIDATION_DIRECTORY, VALIDATION_SHARDS)