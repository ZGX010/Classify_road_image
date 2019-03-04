# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
import cv2
import shutil

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None



class NodeLookup(object):
  def __init__(self, label_lookup_path=None):
    self.node_lookup = self.load(label_lookup_path)

  def load(self, label_lookup_path):
    node_id_to_name = {}
    with open(label_lookup_path) as f:
      for index, line in enumerate(f):
        node_id_to_name[index] = line.strip()
    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def preprocess_for_gry2bgr(image_path):
  image_data = cv2.imread(image_path)
  cv2.imwrite(image_path, image_data)


def run_inference_on_image(image_dir):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  # os.mkdir('./yes')
  # os.mkdir('./no')

  # with tf.Graph().as_default():
  #   preprocess_for_gry2bgr(image)
  #   image_data = tf.gfile.FastGFile(image, 'rb').read()
  #   image_data = tf.image.decode_jpeg(image_data)
  #   image_data = preprocess_for_eval(image_data, 299, 299)
  #   image_data = tf.expand_dims(image_data, 0)
  #   with tf.Session() as sess:
  #     image_data = sess.run(image_data)

  # preprocess_for_gry2bgr(image_dir)
  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('InceptionV4/Logits/Predictions:0')

    for image in os.listdir(image_dir):
      image_path = os.path.join(image_dir,image)
      preprocess_for_gry2bgr(image_path)
      image_data = tf.gfile.FastGFile(image_path, 'rb').read()
      image_data = tf.image.decode_jpeg(image_data)
      image_data = preprocess_for_eval(image_data, 299, 299)
      image_data = tf.expand_dims(image_data, 0)
      image_data = sess.run(image_data)

      # with tf.Graph().as_default():
      #   image_data = sess.run(image_data)
      # with tf.Session() as sess:
      #   image_data = sess.run(image_data)

      predictions = sess.run(softmax_tensor,{'input:0': image_data}) # two
      class_index = np.where(predictions == np.max(predictions, axis=1))
      class_name = class_index[1] #[0]
      # print(image_path,class_name)
      print('%s (class_name = %s)' % (image_path , class_name))
      if class_name == 1 :
        shutil.move(image_path,"./yes/"+image)
        # os.rename(image_path,"./yes/"+image)
      else:
        shutil.move(image_path,"./no/"+image)
        # os.rename(image_path,"./no/"+image)


def main(_):
  image_dir = FLAGS.image_dir
  run_inference_on_image(image_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_path',
      type=str,
  )
  parser.add_argument(
      '--label_path',
      type=str,
  )
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=2,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
