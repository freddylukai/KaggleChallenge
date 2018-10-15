# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Utility functions for creating TFRecord data sets."""

import tensorflow as tf


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_examples_list(path):
  """Read list of training or validation examples.

  The file is assumed to contain a single example per line where the first
  token in the line is an identifier that allows us to find the image and
  annotation xml for that example.

  For example, the line:
  xyz 3
  would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

  Args:
    path: absolute path to examples list file.

  Returns:
    list of example identifiers (strings).
  """
  with tf.gfile.GFile(path) as fid:
    lines = fid.readlines()
  return [line.strip().split(' ')[0] for line in lines]


def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.

  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.

  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

  Returns:
    Python dictionary holding XML contents.
  """
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

from scipy import ndimage
import numpy as np
def rotate_image(img, angle):
  return ndimage.rotate(img, angle, axes=(0,1), mode='reflect', reshape=False)

def shift_image(img, height_shift, width_shift):
  shifted = ndimage.shift(img, shift=(height_shift, width_shift, 0))
  return shifted

def zoom_image(image,height_zoom, width_zoom):
  # Zoom in or out - Make sure pad with zeros or resize for the initial size.
  zoomed = ndimage.zoom(image, (height_zoom, width_zoom, 1))
  return zoomed

def crop_image(image, height, width, channel):
  imgResized = np.zeros((height, width, channel))
  height_small = min(image.shape[0], height)
  width_small = min(image.shape[1], width)
  imgResized[:height_small, :width_small, :] = image[:height_small, :width_small, :]
  return imgResized

def transform_image(image):
  #print('Image shape: ', image.shape)
  angle = np.random.randint(0,45) 
  rotated = rotate_image(image, angle)
  #print('Rotated image shape: ', rotated.shape)
    
  original_shape = image.shape
  height_shift = np.random.randint(0,original_shape[0]/10)
  width_shift = np.random.randint(0, original_shape[1]/10)
  #print("Original Image shape: ", original_shape)
  #print('Shift: {0}, {1}'.format(height_shift, width_shift))
  shifted = shift_image(rotated, height_shift, width_shift)  
  #print('Shifted image shape: ', shifted.shape)

  # Keep the zoom ratio to max of 10% for both zoom in and zoom out.
  #width_zoom = np.random.uniform(0.9,1.1)
  #height_zoom = np.random.uniform(0.9, 1.1)
  #if(width_zoom, height_zoom) == (1,1):
  #  width_zoom, height_zoom = (1.1, 1.1)
  #zoomed = zoom_image(shifted, height_zoom, width_zoom)
  #print('Zoomed image shape: ', zoomed.shape)
    
  #zoomedResized = crop_image(zoomed, original_shape[0], original_shape[1], original_shape[2])
  #print('Zoomed resized image shape: ', zoomedResized.shape)
    
  return shifted
