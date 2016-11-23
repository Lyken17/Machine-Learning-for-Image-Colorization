"""
Config file
"""

from sys import float_info

import tensorflow as tf
from tensorflow.python.ops.image_ops import ResizeMethod


# Debug flag
debug = True

# Epsilon for math calculation
eps = float_info.epsilon

# Image size for training
image_size = 224

# Parameters for neural network
learning_rate = 0.1
training_iters = 1000
batch_size = 6
dequeue_buffer_size = 1000
display_step = 1
test_iters = 10

# Image resize method
input_resize_method = ResizeMethod.BILINEAR
training_resize_method = ResizeMethod.BILINEAR

# Directory for training and testing dataset
train_dir = "train2014"
test_dir = "val2014"

# Weights for each layer
weights = {
    'conv1_1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    'conv1_2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'conv2_1': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'conv2_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'conv3_1': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'conv3_2': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'conv3_3': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'conv4_1': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    'conv4_2': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'conv4_3': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'conv5_1': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'conv5_2': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'conv5_3': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'b_conv4': tf.Variable(tf.random_normal([3, 3, 512, 256])),
    'b_conv3': tf.Variable(tf.random_normal([3, 3, 256, 128])),
    'b_conv2': tf.Variable(tf.random_normal([3, 3, 128, 64])),
    'b_conv1': tf.Variable(tf.random_normal([3, 3, 64, 3])),
    'b_conv0': tf.Variable(tf.random_normal([3, 3, 3, 3])),
}
