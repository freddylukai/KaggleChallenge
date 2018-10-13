import pandas as pd
import os
import numpy as np
from tensorflow.python.lib.io import file_io

PNEUMONIA_LABEL = 'pneumonia'

# Probability of image to be set as test instead of training
TEST_PROB = 0.4

'''
    input_path: folder with the following structure:
        stage_1_train_images/
        stage_1_train_labels.csv

        returns all_data, classes_count, class_mapping
        
    note: adding an additional class 'bg' may help. It represents a label which
    is the background.
'''
def get_data(input_path):
    global PNEUMONIA_LABEL

    train_labels_csv_path = os.path.join(input_path, 'stage_1_train_labels.csv')
    train_images_folder_path = os.path.join(input_path, 'stage_1_train_images')
    train_images_folder_path = os.path.join(train_images_folder_path, 'jpgs')

    with file_io.FileIO(train_labels_csv_path, 'r') as f:
        df = pd.read_csv(f)

    counter = 0
    parsed = {}

    for _, row in df.iterrows():

        pid = row['patientId']
        filename =  os.path.join(train_images_folder_path, '%s.jpeg' % pid)
        counter += 1

        if filename not in parsed:
            parsed[pid] = {
                'filepath': filename,
                'bboxes': [],
                'width': 1024,
                'height': 1024,
                'imageset': 'test' if np.random.random() <= TEST_PROB else 'trainval'
            }

        if row['Target'] == 1:
            x1, y1 = row['x'], row['y']
            x2, y2, = x1 + row['width'], y1 + row['height']
            parsed[pid]['bboxes'].append({'class': PNEUMONIA_LABEL, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})



    # Returns list of data, mapping from label to number, and the count of labels (only one in our case)
    return list(parsed.values()), {PNEUMONIA_LABEL: counter}, {PNEUMONIA_LABEL: 0}