"""
Residual-encoder model implementation.

See extensive documentation at
http://tinyclouds.org/colorize/
"""

import os

from tensorflow.contrib.layers import batch_norm

from color_space_convert import rgb_to_yuv
from config import *


def init_file_path(directory):
    """
    Get the image file path array
    :param directory: the directory that store images
    :return: an array of image file path
    """
    paths = []
    for file in os.listdir(directory):
        # Skip files that is not jpg
        if not file.endswith('.jpg'):
            continue
        file_path = '%s/%s' % (directory, file)
        paths.append(file_path)
    return paths


def read_image(file_path):
    """
    Read and store image with YUV color space
    :param file_path: the path for the image file
    :return: image with YUV color space
    """
    # Read the image with RGB color space
    rgb_image = tf.image.decode_jpeg(tf.read_file(file_path), channels=3)
    # Make pixel element value in [0, 1)
    rgb_image = tf.image.convert_image_dtype(rgb_image, tf.float32)
    # Resize image to the right image_size
    rgb_image = tf.image.resize_images(rgb_image, [image_size, image_size], method=input_resize_method)
    # Change color space to YUV
    yuv_image = rgb_to_yuv(rgb_image)
    return yuv_image


def get_y_and_uv_data(file_paths):
    """
    Get the input data with Y channel and target data with UV channels
    :param file_paths: the path for the image file
    :return: input data with Y channel and target data with UV channels
    """
    _y = []
    _uv = []
    for i in file_paths:
        # Get the image with YUV channels
        t = read_image(i)
        # Split channels
        channel_y = tf.slice(t, [0, 0, 0], [image_size, image_size, 1])
        channel_u = tf.slice(t, [0, 0, 1], [image_size, image_size, 1])
        channel_v = tf.slice(t, [0, 0, 2], [image_size, image_size, 1])
        # Normalize channels
        channel_y = tf.mul(tf.sub(channel_y, 0.5), 2.0)
        channel_u = tf.div(channel_u, 0.436)
        channel_v = tf.div(channel_v, 0.615)
        # Add channel data
        _y.append(channel_y)
        _uv.append(tf.pack([channel_u, channel_v], axis=0))
        break
    return _y, _uv


class ResidualEncoder(object):
    def __init__(self, train_input, train_output):
        self.train_input = train_input
        self.train_output = train_output
        self.data_size = tf.shape(train_input)[0]
        self.counter = 0

    @staticmethod
    def get_weight(name):
        """
        Get weight for one layer
        :param name: the name of the layer
        :return: the initial weight for this layer
        """
        return weights[name]

    @staticmethod
    def get_bias(name):
        """
        Get bias for one layer
        :param name: the name of the layer
        :return: the initial bias for this layer
        """
        return biases[name]

    @staticmethod
    def get_cost(predict_val, real_val):
        # TODO: Implement this function
        return 0

    def get_next_batch(self, b_size):
        """
        Get next batch of data
        :param b_size: the size of mini batch
        :return: a batch of data
        """
        _input = []
        _output = []
        for i in range(b_size):
            self.counter += 1
            if self.counter >= self.data_size:
                self.counter = 0
            _input.append(self.train_input[i])
            _output.append(self.train_output[i])
        _input = tf.pack(_input, axis=0)
        _output = tf.pack(_output, axis=0)
        return _input, _output

    def conv_layer(self, layer_input, name):
        """
        Convolution layer
        :param layer_input: the input data for this layer
        :param name: name for this layer
        :return: the layer data after convolution
        """
        with tf.variable_scope(name):
            weight = self.get_weight(name)
            conv_biases = self.get_bias(name)
            output = tf.nn.conv2d(layer_input, weight, strides=[1, 1, 1, 1], padding='SAME')
            # output = tf.nn.bias_add(output, conv_biases)
            # output = tf.nn.relu(output)
            output = self.batch_normal(output, is_training=is_training, scope=name)
            return output

    def max_pool(self, layer_input, name):
        """
        Polling layer
        :param layer_input: the input data for this layer
        :param name: name for this layer
        :return: the layer data after max-pooling
        """
        return tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def batch_normal(self, input_data, is_training, scope):
        return tf.cond(is_training,
                       lambda: batch_norm(input_data, decay=0.9, is_training=True, center=True, scale=True,
                                          activation_fn=tf.nn.relu, updates_collections=None, scope=scope),
                       lambda: batch_norm(input_data, decay=0.9, is_training=False, center=True, scale=True,
                                          activation_fn=tf.nn.relu, updates_collections=None, scope=scope, reuse=True))

    def build(self, input_data):
        """
        Build the residual encoder model
        :param input_data: input data for first layer
        :return: None
        """
        if debug:
            assert input_data.get_shape().as_list()[1:] == [224, 224, 1]

        # Make channel duplicated to 3 channels
        input_data = tf.image.grayscale_to_rgb(input_data)

        if debug:
            assert input_data.get_shape().as_list()[1:] == [224, 224, 3]
            assert input_data[0][1][1][1] == input_data[0][1][1][2]
            assert input_data[0][1][1][1] == input_data[0][1][1][3]
            assert input_data[0][100][100][1] == input_data[0][100][100][2]
            assert input_data[0][100][100][1] == input_data[0][100][100][3]

        # First convolutional layer
        conv1_1 = self.conv_layer(input_data, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        if debug:
            assert conv1_1.get_shape().as_list()[1:] == [224, 224, 64]
            assert conv1_2.get_shape().as_list()[1:] == [224, 224, 64]
            assert pool1.get_shape().as_list()[1:] == [112, 112, 64]

        # Second convolutional layer
        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        if debug:
            assert conv2_1.get_shape().as_list()[1:] == [112, 112, 128]
            assert conv2_2.get_shape().as_list()[1:] == [112, 112, 128]
            assert pool2.get_shape().as_list()[1:] == [56, 56, 128]

        # Third convolutional layer
        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        if debug:
            assert conv3_1.get_shape().as_list()[1:] == [56, 56, 256]
            assert conv3_2.get_shape().as_list()[1:] == [56, 56, 256]
            assert conv3_3.get_shape().as_list()[1:] == [56, 56, 256]
            assert pool3.get_shape().as_list()[1:] == [28, 28, 256]

        # Fourth convolutional layer
        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        if debug:
            assert conv4_1.get_shape().as_list()[1:] == [28, 28, 512]
            assert conv4_2.get_shape().as_list()[1:] == [28, 28, 512]
            assert conv4_3.get_shape().as_list()[1:] == [28, 28, 512]
            assert pool4.get_shape().as_list()[1:] == [14, 14, 512]

        # Fifth convolutional layer
        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")

        if debug:
            assert conv5_1.get_shape().as_list()[1:] == [14, 14, 512]
            assert conv5_2.get_shape().as_list()[1:] == [14, 14, 512]
            assert conv5_3.get_shape().as_list()[1:] == [14, 14, 512]

        # Backward upscale and convolutional layers
        b_conv5_upscale = tf.image.resize_images(conv5_3, [28, 28, 512], method=training_resize_method)
        b_conv4 = self.conv_layer(tf.add(conv4_3, b_conv5_upscale), "b_conv4")
        b_conv4_upscale = tf.image.resize_images(b_conv4, [56, 56, 256], method=training_resize_method)
        b_conv3 = self.conv_layer(tf.add(conv3_3, b_conv4_upscale), "b_conv3")
        b_conv3_upscale = tf.image.resize_images(b_conv3, [112, 112, 128], method=training_resize_method)
        b_conv2 = self.conv_layer(tf.add(conv2_2, b_conv3_upscale), "b_conv2")
        b_conv2_upscale = tf.image.resize_images(b_conv2, [224, 224, 64], method=training_resize_method)
        b_conv1 = self.conv_layer(tf.add(conv1_2, b_conv2_upscale), "b_conv1")
        output_layer = self.conv_layer(tf.add(input_data, b_conv1), "output_layer")

        return output_layer


if __name__ == '__main__':
    # Init image data file path
    train_file_paths = init_file_path(train_dir)
    test_file_paths = init_file_path(test_dir)

    # Get YUV image data
    train_y, train_uv = get_y_and_uv_data(train_file_paths)
    test_y, test_uv = get_y_and_uv_data(test_file_paths)

    # Init training and testing data
    train_y = tf.pack(train_y, axis=0)
    train_uv = tf.pack(train_uv, axis=0)
    test_y = tf.pack(test_y, axis=0)
    test_uv = tf.pack(test_uv, axis=0)

    # Init placeholder for input and output
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
    y = tf.placeholder(tf.float32, [None, image_size, image_size, 2])
    is_training = tf.placeholder(tf.bool)

    # Init residual encoder model
    residual_encoder = ResidualEncoder(train_y, train_uv)

    # Make predict, cost and training model
    predict = residual_encoder.build(input_data=x)
    cost = residual_encoder.get_cost(predict_val=predict, real_val=y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # TODO: Implement accuracy function
    accuracy = 0

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        # Start training
        print "Start training!!!"
        step = 1
        while step * batch_size < training_iters:
            batch_xs, batch_ys = residual_encoder.get_next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, is_training: True})
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, is_training: True})
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, is_training: True})
                print "Iter %d, Minibatch Loss = %f, Training Accuracy = %f" % (step, loss, acc)
                step += 1
        print "Training Finished!"

        # Predict
        # TODO: Test with testing data
        # print "Testing Accuracy: %f" % (sess.run(accuracy, feed_dict={x: 0, y: 0, is_training: False}))
