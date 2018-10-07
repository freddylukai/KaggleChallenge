import cv2
import os
import pydicom
import numpy as np
import scipy.misc

'''
    Wrapper to open an image to support dcm files.
'''
def imread(filename):
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.dcm':
        img = pydicom.read_file(filename).pixel_array
        return np.stack((img,) * 3, -1)
    else:
        return cv2.imread(filename)
