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
learning_rate = 0.0000001
training_iters = 1000
batch_size = 30
dequeue_buffer_size = 1000
display_step = 1
save_step = 50
test_iters = 100

# Image resize method
input_resize_method = ResizeMethod.BILINEAR
training_resize_method = ResizeMethod.BILINEAR

# YUV normalization parameters
y_norm_para = 0.5
u_norm_para = 0.436
v_norm_para = 0.615

# Directory for training and testing dataset
train_dir = "train2014"
test_dir = "val2014"

# Weights for each layer
weights = {
    'b_conv4': tf.Variable(tf.random_normal([1, 1, 512, 256], stddev=0.01)),
    'b_conv3': tf.Variable(tf.random_normal([3, 3, 256, 128], stddev=0.01)),
    'b_conv2': tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=0.01)),
    'b_conv1': tf.Variable(tf.random_normal([3, 3, 64, 3], stddev=0.01)),
    'b_conv0': tf.Variable(tf.random_normal([3, 3, 3, 3], stddev=0.01)),
    'output_conv': tf.Variable(tf.random_normal([3, 3, 3, 2], stddev=0.01)),
}

# Biases for each layer
biases = {
    'b_conv4': tf.Variable(tf.random_normal([256], stddev=0.01)),
    'b_conv3': tf.Variable(tf.random_normal([128], stddev=0.01)),
    'b_conv2': tf.Variable(tf.random_normal([64], stddev=0.01)),
    'b_conv1': tf.Variable(tf.random_normal([3], stddev=0.01)),
    'b_conv0': tf.Variable(tf.random_normal([3], stddev=0.01)),
    'output_conv': tf.Variable(tf.random_normal([2], stddev=0.01)),
}
