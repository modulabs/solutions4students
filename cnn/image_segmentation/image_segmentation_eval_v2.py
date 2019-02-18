#!/usr/bin/env python
# coding: utf-8

# # Image Segmentation for Evaluation
# 
# * MeanIOU: Image Segmentation에서 많이 쓰이는 evaluation measure
# * tf.version 1.12 API: [`tf.metrics.mean_iou`](https://www.tensorflow.org/api_docs/python/tf/metrics/mean_iou)
#   * `tf.enable_eager_execution()`이 작동하지 않음
#   * 따라서 예전 방식대로 `tf.Session()`을 이용하여 작성하거나 아래와 같이 2.0 version으로 작성하여야 함
# * tf.version 2.0 API: [`tf.keras.metrics.MeanIoU`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/metrics/MeanIoU)
# * 지금 이 코드는 `version 2` 코드를 이용하여 작성

# ## Import modules

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

from sklearn.model_selection import train_test_split
from PIL import Image
from IPython.display import clear_output

import tensorflow as tf

from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# ## Load test data

# In[ ]:


dataset_dir = '../../datasets/sd_train'
img_dir = os.path.join(dataset_dir, "train")
label_dir = os.path.join(dataset_dir, "train_labels")


# In[ ]:


x_train_filenames = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
x_train_filenames.sort()
y_train_filenames = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir)]
y_train_filenames.sort()


# In[ ]:


x_train_filenames, x_test_filenames, y_train_filenames, y_test_filenames =                     train_test_split(x_train_filenames, y_train_filenames, test_size=0.2, random_state=219)


# In[ ]:


num_train_examples = len(x_train_filenames)
num_test_examples = len(x_test_filenames)

print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples: {}".format(num_test_examples))


# ## Build our input pipeline with `tf.data`
# ### Set up test datasets

# In[ ]:


# Set hyperparameters
image_size = 64
img_shape = (image_size, image_size, 3)
batch_size = 60 # all test dataset


# In[ ]:


def _process_pathnames(fname, label_path):
  # We map this function onto each pathname pair
  img_str = tf.io.read_file(fname)
  img = tf.image.decode_bmp(img_str, channels=3)

  label_img_str = tf.io.read_file(label_path)
  label_img = tf.image.decode_bmp(label_img_str, channels=1)
  
  resize = [image_size, image_size]
  img = tf.image.resize(img, resize)
  label_img = tf.image.resize(label_img, resize)
  
  scale = 1 / 255.
  img = tf.dtypes.cast(img, tf.float32) * scale
  label_img = tf.dtypes.cast(label_img, tf.float32) * scale
  
  return img, label_img


# In[ ]:


def get_baseline_dataset(filenames,
                         labels,
                         threads=5,
                         batch_size=batch_size,
                         shuffle=True):
  num_x = len(filenames)
  # Create a dataset from the filenames and labels
  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  # Map our preprocessing function to every element in our dataset, taking
  # advantage of multithreading
  dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
  
  if shuffle:
    dataset = dataset.shuffle(num_x * 10)
  
  dataset = dataset.batch(batch_size)
  return dataset


# In[ ]:


test_dataset = get_baseline_dataset(x_test_filenames,
                                    y_test_filenames,
                                    shuffle=False)


# ## Build the model

# In[ ]:


def conv_block(input_tensor, num_filters):
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  return encoder

def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  
  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  return decoder


# In[ ]:


inputs = layers.Input(shape=img_shape)
# 256

encoder0_pool, encoder0 = encoder_block(inputs, 32) # 128
encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64) # 64
encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128) # 32
encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16

center = conv_block(encoder3_pool, 512) # center

decoder3 = decoder_block(center, encoder3, 256) # 32
decoder2 = decoder_block(decoder3, encoder2, 128) # 64
decoder1 = decoder_block(decoder2, encoder1, 64) # 128
decoder0 = decoder_block(decoder1, encoder0, 32) # 256

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)


# In[ ]:


model = models.Model(inputs=[inputs], outputs=[outputs])


# ## Restore using Checkpoints (Object-based saving)

# In[ ]:


checkpoint_dir = 'train/exp1'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model)

# Restore the latest checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# ### Display prediction mask image for one test data

# In[ ]:


for test_images, test_labels in test_dataset.take(1):
  predictions = model(test_images)
        
  plt.figure(figsize=(10, 20))
  plt.subplot(1, 3, 1)
  plt.imshow(test_images[0,: , :, :])
  plt.title("Input image")

  plt.subplot(1, 3, 2)
  plt.imshow(test_labels[0, :, :, 0])
  plt.title("Actual Mask")

  plt.subplot(1, 3, 3)
  plt.imshow(predictions[0, :, :, 0])
  plt.title("Predicted Mask")
  plt.show()


# ## Evaluate the test dataset and Plot

# In[ ]:


m = tf.keras.metrics.MeanIoU(num_classes=2)

for images, labels in test_dataset:
  predictions = model(images)
  m.update_state( tf.dtypes.cast(tf.math.round(labels), tf.int32),
                  tf.dtypes.cast(tf.math.round(predictions), tf.int32) )

print('Final Mean IOU result: ', m.result().numpy())

