import cv2
import os
import pydicom
import numpy as np
import scipy.misc
from tensorflow.python.lib.io import file_io
'''
    Wrapper to open an image to support dcm files.
'''
def imread(filename):
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.dcm':
        with file_io.FileIO(filename, 'rb') as f:
            img = pydicom.read_file(f).pixel_array
            return np.stack((img,) * 3, -1)
    else:
        model_file = file_io.FileIO(filename, mode='rb')
        temp_model_location = './temp.jpeg'
        temp_model_file = open(temp_model_location, 'wb')
        temp_model_file.write(model_file.read())
        temp_model_file.close()
        return cv2.imread(temp_model_location)
