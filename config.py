"""
Config file for residual-encoder model
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
training_iters = 500
batch_size = 10
display_step = 1

# Parameters for batch normalization
bn_mean = 0.0
bn_variance = 1.0
bn_offset = None
bn_scale = None

# Image resize method
input_resize_method = ResizeMethod.NEAREST_NEIGHBOR
training_resize_method = ResizeMethod.NEAREST_NEIGHBOR

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
}

# Biases for each layer
biases = {
    'conv1_1': tf.Variable(tf.random_normal([64])),
    'conv1_2': tf.Variable(tf.random_normal([64])),
    'conv2_1': tf.Variable(tf.random_normal([128])),
    'conv2_2': tf.Variable(tf.random_normal([128])),
    'conv3_1': tf.Variable(tf.random_normal([256])),
    'conv3_2': tf.Variable(tf.random_normal([256])),
    'conv3_3': tf.Variable(tf.random_normal([256])),
    'conv4_1': tf.Variable(tf.random_normal([512])),
    'conv4_2': tf.Variable(tf.random_normal([512])),
    'conv4_3': tf.Variable(tf.random_normal([512])),
    'conv5_1': tf.Variable(tf.random_normal([512])),
    'conv5_2': tf.Variable(tf.random_normal([512])),
    'conv5_3': tf.Variable(tf.random_normal([512])),
}
