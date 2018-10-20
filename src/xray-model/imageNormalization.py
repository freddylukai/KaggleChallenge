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
import numpy as np


TRAINING_SHARDS = 1024
VALIDATION_SHARDS = 128
TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--gcsdir", dest="gcs_dir", help="GCS base dir")
    parser.add_option("-i", "--imgdir", dest="img_dir", help="PNNGs dir")
    #parser.add_option("-t", "--tfrecorddir", dest="tfrecord_dir", help="tfrecords dir")

    (options, args) = parser.parse_args()

    if not options.gcs_dir:
        raise Exception('GCS dir not provided')

    if not options.img_dir:
        raise Exception('img_dir not provided')

    #if not options.tfrecord_dir:
    #    raise Exception('tfrecord_dir not provided')


    with file_io.FileIO(os.path.join(options.gcs_dir, 'Data_Entry_2017.csv'), 'r') as f:
        dataentry = pd.read_csv(f)

    with file_io.FileIO(os.path.join(options.gcs_dir, 'ClassesLabel.csv'), 'r') as f:
        mapping_df = pd.read_csv(f)

    with file_io.FileIO(os.path.join(options.gcs_dir, 'split.csv'), 'r') as f:
        train_map_df = pd.read_csv(f)

    #tf_records_path = os.path.join(options.gcs_dir, 'records')

    allowed_images = set([x['Image Index'] for _, x in dataentry.iterrows() if '|' not in x['Finding Labels']])
    #allowed_images = set(list(os.walk(options.img_dir))[0][2])

    image_to_class = {x['Image Index']: x['Finding Labels'] for _, x in dataentry.iterrows() if '|' not in x['Finding Labels']}
    class_to_label = {x['Classes']: x['Label'] for _, x in mapping_df.iterrows()}

    training_names = [x['image'] for _, x in train_map_df.iterrows() if x['train'] == 1 and x['image'] in allowed_images]
    validation_names = [x['image'] for _, x in train_map_df.iterrows() if x['train'] == 0 and x['image'] in allowed_images]
    training_classes = [image_to_class[x] for x in training_names]
    validation_classes = [image_to_class[x] for x in validation_names]

    training_filenames = [os.path.join(options.img_dir, x) for x in training_names]
    validation_filenames = [os.path.join(options.img_dir, x) for x in validation_names]

    # Get the average of all training images per channel
    counter = 0
    rMeanGlobal = 0.0
    gMeanGlobal = 0.0
    bMeanGlobal = 0.0
    rStdGlobal = 0.0
    gStdGlobal = 0.0
    bStdGlobal = 0.0

    for imageFile in training_filenames:
        model_file = file_io.FileIO(os.path.join(options.img_dir,imageFile), mode='rb')
        temp_model_location = './temp.png'
        temp_model_file = open(temp_model_location, 'wb')
        temp_model_file.write(model_file.read())
        temp_model_file.close()

        img = cv2.imread(temp_model_location)
        rMean = np.mean(img[:, :, 0])
        gMean = np.mean(img[:, :, 1])
        bMean = np.mean(img[:, :, 2])
        rStd = np.std(img[:, :, 0])
        gStd = np.std(img[:, :, 1])
        bStd = np.std(img[:, :, 2])
        counter += 1
        rMeanGlobal = (rMeanGlobal * (counter-1) + rMean)/(counter)
        gMeanGlobal = (gMeanGlobal * (counter-1) + gMean)/(counter)
        bMeanGlobal = (bMeanGlobal * (counter-1) + bMean)/(counter)
        rStdGlobal = (rStdGlobal * (counter-1) + rStd)/(counter)
        gStdGlobal = (gStdGlobal * (counter-1) + gStd)/(counter)
        bStdGlobal = (bStdGlobal * (counter-1) + bStd)/(counter)
        if counter % 100 == 0:
            print('Image Count: {}, Mean:({},{},{}), Std: ({},{},{})'.format(counter,rMeanGlobal,gMeanGlobal,bMeanGlobal,rStdGlobal,gStdGlobal,bStdGlobal))

    print(
        'Final result: Image Count: {}, Mean:({},{},{}), Std: ({},{},{})'.format(counter, rMeanGlobal, gMeanGlobal, bMeanGlobal, rStdGlobal,
                                                             gStdGlobal, bStdGlobal))
