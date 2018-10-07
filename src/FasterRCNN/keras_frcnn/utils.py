import cv2
import os
import pydicom

'''
    Wrapper to open an image to support dcm files.
'''
def imread(filename):
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.dcm':
        return pydicom.read_file(filename).pixel_array
    else:
        return cv2.imread(filename)
