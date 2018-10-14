import tensorflow as tf
import sys
from optparse import OptionParser
import os
from tensorflow.python.lib.io import file_io
import zipfile
import cv2
import shutil

#zip_filenames = ['images_001.zip', 'images_002.zip', 'images_003.zip', 'images_004.zip', 'images_005.zip', 'images_006.zip', 'images_007.zip', 'images_008.zip', 'images_009.zip', 'images_010.zip', 'images_011.zip', 'images_012.zip']
zip_filenames = ['images_001.zip']

temp_model_location = './temp.zip'
extract_dir = './temp_extract_dir'

def prepare_folder(folder_path):
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)

    os.mkdir(folder_path)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--gcsdir", dest="gcs_dir", help="GCS base dir")

    (options, args) = parser.parse_args()

    if not options.gcs_dir:
        raise Exception('GCS dir not provided')


    for zip_filename in zip_filenames:
        prepare_folder(extract_dir)

        gcs_filepath = os.path.join(options.gcs_dir, zip_filename)
        with file_io.FileIO(gcs_filepath, mode='rb') as model_file:
            with open(temp_model_location, 'wb') as temp_model_file:
                temp_model_file.write(model_file.read())


        with zipfile.ZipFile(temp_model_location, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        images_folder = os.path.join(extract_dir, 'images')
        for filename in os.listdir(images_folder):
            full_filename = os.path.join(images_folder, filename)

            if os.path.isfile(full_filename) and full_filename.endswith('.png'):
                img = cv2.imread(full_filename)
                print(full_filename, img.shape)