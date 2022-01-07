# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Reader functions for Cityscapes/KITTI struct2depth data.

  This implements the interface functions for reading Cityscapes data.

  Each image file consists of 3 consecutive frames concatenated along the width
  dimension, stored in png format. The camera intrinsics are stored in a file
  that has the same name as the image, with a 'txt' extension and the
  coefficients flattened inside. The 'train.txt' file lists the training
  samples.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.compat.v1 as tf
import parameter_container
import numpy as np


FORMAT_NAME = 'STRUCT2DEPTH'
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGES_PER_SEQUENCE = 3
CHANNELS = 3
K1 = np.array([1998.7356 , 0, 851.8287, 0, 1991.8909 , 424.0041, 0, 0, 1], dtype=np.float32)


READ_FRAME_PAIRS_FROM_DATA_PATH_PARAMS = {
    # Number of parallel threads for reading.
    'num_parallel_calls': 64,
}


def read_frame_pairs_from_data_path(train_file_path, params=None):
  """Reads frame pairs from a text file in the struct2depth format.

  Args:
    train_file_path: A string, file path to the text file listing the training
      examples.
    params: A dictionary or a ParameterContainer with overrides for the default
      params (READ_FRAME_PAIRS_FROM_DATA_PATH_PARAMS)

  Returns:
    A dataset object.
  """
  return read_frame_sequence_from_data_path(
      train_file_path, sequence_length=2, params=params)


def make_filename(lines, directory):
  img_pair_list = list()

  for ii, line in enumerate(lines):
    img_info = line.split()

    if len(img_info) == 1:   # raw fps
      frame_0 = os.path.join(directory,lines[ii-1].split()[0])
      frame_1 = os.path.join(directory,img_info[0])
      img_pair_list.append(frame_0 + ' ' + frame_1)
   
  return img_pair_list



def read_frame_sequence_from_data_path(train_file_path,
                                       sequence_length=IMAGES_PER_SEQUENCE,
                                       params=None):
  """Reads frames sequences from a text file in the struct2depth format.

  Args:
    train_file_path: A string, file path to the text file listing the training
      examples.
    sequence_length: Number of images in the output sequence (1, 2, or 3).
    params: A dictionary or a ParameterContainer with overrides for the default
      params (READ_FRAME_PAIRS_FROM_DATA_PATH_PARAMS)

  Returns:
    A dataset object.
  """
  if sequence_length not in (1, 2, 3):
    raise ValueError('sequence_length must be in (1, 2, 3), not %d.' %
                     sequence_length)
  params = parameter_container.ParameterContainer(
      READ_FRAME_PAIRS_FROM_DATA_PATH_PARAMS, params)

  with tf.gfile.Open(train_file_path) as f:
    lines = f.read().split('\n')
  lines = list(filter(None, lines))  # Filter out empty strings.
  directory = os.path.dirname(train_file_path)

  files = make_filename(lines, directory)
  ds = tf.data.Dataset.from_tensor_slices(files)
  ds = ds.repeat()

  def parse_fn_for_pairs(filename):
    return parse_fn(filename, output_sequence_length=sequence_length)

  num_parallel_calls = min(len(files), params.num_parallel_calls)
  ds = ds.map(parse_fn_for_pairs, num_parallel_calls=num_parallel_calls)

  return ds


def parse_fn(filename,
             output_sequence_length=IMAGES_PER_SEQUENCE):
  """Read data from single files stored in directories.

  Args:
    filename: the filename of the set of files to be loaded.
    output_sequence_length: Length of the output sequence. If less than
      IMAGES_PER_SEQUENCE, only the first `output_sequence_length` frames will
      be kept.

  Returns:
    A dictionary that maps strings to tf.Tensors of type float32:

    'rgb': an RGB image of shape H, W, 3. Each channel value is between 0.0 and
           1.0.
    'intrinsics': a list of intrinsics values.
  """
  if output_sequence_length > IMAGES_PER_SEQUENCE or output_sequence_length < 1:
    raise ValueError('Invalid output_sequence_length %d: must be within [1, '
                     '%d].' % (output_sequence_length, IMAGES_PER_SEQUENCE))
  
  image_file_1 = tf.strings.split([filename]).values[0]
  image_file_2 = tf.strings.split([filename]).values[1]
 
  # Read files.
  encoded_image_1 = tf.io.read_file(image_file_1) 
  encoded_image_2 = tf.io.read_file(image_file_2)
  
  # Parse intrinsics data to a tensor representing a 3x3 matrix.
  intrinsics = tf.convert_to_tensor(K1)
  intrinsics.set_shape([9])

  fx, _, x0, _, fy, y0, _, _, _ = tf.unstack(intrinsics)
  intrinsics = tf.stack([IMAGE_WIDTH, IMAGE_HEIGHT, fx, fy, x0, y0])

  # Decode and normalize images.
  decoded_image_1 = tf.image.decode_jpeg(encoded_image_1, channels=3)
  decoded_image_1 = tf.to_float(decoded_image_1) * (1 / 255.0)
  decoded_image_2 = tf.image.decode_jpeg(encoded_image_2, channels=3)
  decoded_image_2 = tf.to_float(decoded_image_2) * (1 / 255.0)


  return {
      'rgb': tf.stack([decoded_image_1,decoded_image_2], axis=0),
      'intrinsics': tf.stack([intrinsics] * output_sequence_length),
  }


def read_and_parse_data(files):
  """Default reader and parser for reading Cityscapes/KITTI data from files.

  Args:
    files: a list of filenames. Each filename is extended to image and camera
      intrinsics filenames in parse_fn.

  Returns:
    A preprocessing function representing data stored as a collection of files.
  """
  ds = tf.data.Dataset.from_tensor_slices(files)
  ds = ds.repeat()
  ds = ds.map(parse_fn)

  return ds


def main(params):
  """An Estimator's input_fn for reading and preprocessing training data.

  Reads pairs of RGBD frames from sstables, filters out near duplicates and
  performs data augmentation.

  Args:
    params: A dictionary with hyperparameters.

  Returns:
    A tf.data.Dataset object.
  """

  params = parameter_container.ParameterContainer.from_defaults_and_overrides(
      DEFAULT_PARAMS, params, is_strict=True, strictness_depth=2)
  dataset = read_frame_pairs_from_data_path(
      params.input.data_path, params.input.reader)

  if params.learn_intrinsics.enabled and params.learn_intrinsics.per_video:
    intrinsics_ht = intrinsics_utils.HashTableIndexer(
        params.learn_intrinsics.max_number_of_videos)

  def key_to_index(input_endpoints):
    video_id = input_endpoints.pop('video_id', None)
    if (video_id is not None and params.learn_intrinsics.enabled and
        params.learn_intrinsics.per_video):
      index = intrinsics_ht.get_or_create_index(video_id[0])
      input_endpoints['video_index'] = index
      input_endpoints['video_index'] = tf.stack([index] * 2)
    return input_endpoints

  dataset = dataset.map(key_to_index)

  def is_duplicate(endpoints):
    """Implements a simple duplicate filter, based on L1 difference in RGB."""
    return tf.greater(
        tf.reduce_mean(tf.abs(endpoints['rgb'][1] - endpoints['rgb'][0])),
        params.input.duplicates_filter_threshold)

  if params.input.duplicates_filter_threshold > 0.0:
    dataset = dataset.filter(is_duplicate)

  # Add data augmentation
  if params.image_preprocessing.data_augmentation:
    if params.learn_intrinsics.per_video:
      raise ('Data augemnation together with learn_intrinsics.per_video is not '
             'yet supported.')

    def random_crop_and_resize_fn(endpoints):
      return data_processing.random_crop_and_resize_pipeline(
          endpoints, params.image_preprocessing.image_height,
          params.image_preprocessing.image_width)

    augmentation_fn = random_crop_and_resize_fn
  else:
    def resize_fn(endpoints):
      return data_processing.resize_pipeline(
          endpoints, params.image_preprocessing.image_height,
          params.image_preprocessing.image_width)

    augmentation_fn = resize_fn

  dataset = dataset.map(augmentation_fn)
  dataset = dataset.shuffle(params.input.shuffle_queue_size)
  dataset = dataset.batch(params.batch_size, drop_remainder=True)

  return dataset.prefetch(params.input.prefetch_size)
