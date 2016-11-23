"""
Residual-encoder model implementation.

See extensive documentation at
http://tinyclouds.org/colorize/
"""

from tensorflow.contrib.layers import batch_norm

from config import *


class ResidualEncoder(object):
    def __init__(self):
        pass

    @staticmethod
    def get_weight(name):
        """
        Get weight for one layer
        :param name: the name of the layer
        :return: the initial weight for this layer
        """
        return weights[name]

    @staticmethod
    def get_cost(predict_val, real_val):
        """
        Cost function
        :param predict_val: the predict value
        :param real_val: the real value
        :return: cost
        """
        diff = tf.sub(predict_val, real_val, name="diff")
        square = tf.square(diff, name="square")
        return tf.div(tf.reduce_mean(square, name="cost"), batch_size, name="cost_per_batch")

    @staticmethod
    def batch_normal(input_data, training_flag, scope):
        return tf.cond(training_flag,
                       lambda: batch_norm(input_data, decay=0.9, is_training=True, center=True, scale=True,
                                          activation_fn=tf.nn.relu, updates_collections=None, scope=scope),
                       lambda: batch_norm(input_data, decay=0.9, is_training=False, center=True, scale=True,
                                          activation_fn=tf.nn.relu, updates_collections=None, scope=scope, reuse=True), name='batch_normalization')

    @staticmethod
    def max_pool(layer_input, name):
        """
        Polling layer
        :param layer_input: the input data for this layer
        :param name: name for this layer
        :return: the layer data after max-pooling
        """
        return tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, layer_input, name, is_training):
        """
        Convolution layer
        :param layer_input: the input data for this layer
        :param name: name for this layer
        :param is_training: a flag indicate if now is in training
        :return: the layer data after convolution
        """
        with tf.variable_scope(name):
            weight = self.get_weight(name)
            output = tf.nn.conv2d(layer_input, weight, strides=[1, 1, 1, 1], padding='SAME')
            output = self.batch_normal(output, training_flag=is_training, scope=name)
            return output

    def build(self, input_data, is_training):
        """
        Build the residual encoder model
        :param input_data: input data for first layer
        :param is_training: a flag indicate if now is in training
        :return: None
        """
        if debug:
            assert input_data.get_shape().as_list()[1:] == [224, 224, 1]

        # Make channel duplicated to 3 channels
        input_data = tf.concat(concat_dim=3, values=[input_data, input_data, input_data])

        if debug:
            assert input_data.get_shape().as_list()[1:] == [224, 224, 3]

        # First convolutional layer
        conv1_1 = self.conv_layer(input_data, "conv1_1", is_training)
        conv1_2 = self.conv_layer(conv1_1, "conv1_2", is_training)
        pool1 = self.max_pool(conv1_2, 'pool1')

        if debug:
            assert conv1_1.get_shape().as_list()[1:] == [224, 224, 64]
            assert conv1_2.get_shape().as_list()[1:] == [224, 224, 64]
            assert pool1.get_shape().as_list()[1:] == [112, 112, 64]

        # Second convolutional layer
        conv2_1 = self.conv_layer(pool1, "conv2_1", is_training)
        conv2_2 = self.conv_layer(conv2_1, "conv2_2", is_training)
        pool2 = self.max_pool(conv2_2, 'pool2')

        if debug:
            assert conv2_1.get_shape().as_list()[1:] == [112, 112, 128]
            assert conv2_2.get_shape().as_list()[1:] == [112, 112, 128]
            assert pool2.get_shape().as_list()[1:] == [56, 56, 128]

        # Third convolutional layer
        conv3_1 = self.conv_layer(pool2, "conv3_1", is_training)
        conv3_2 = self.conv_layer(conv3_1, "conv3_2", is_training)
        conv3_3 = self.conv_layer(conv3_2, "conv3_3", is_training)
        pool3 = self.max_pool(conv3_3, 'pool3')

        if debug:
            assert conv3_1.get_shape().as_list()[1:] == [56, 56, 256]
            assert conv3_2.get_shape().as_list()[1:] == [56, 56, 256]
            assert conv3_3.get_shape().as_list()[1:] == [56, 56, 256]
            assert pool3.get_shape().as_list()[1:] == [28, 28, 256]

        # Fourth convolutional layer
        conv4_1 = self.conv_layer(pool3, "conv4_1", is_training)
        conv4_2 = self.conv_layer(conv4_1, "conv4_2", is_training)
        conv4_3 = self.conv_layer(conv4_2, "conv4_3", is_training)
        pool4 = self.max_pool(conv4_3, 'pool4')

        if debug:
            assert conv4_1.get_shape().as_list()[1:] == [28, 28, 512]
            assert conv4_2.get_shape().as_list()[1:] == [28, 28, 512]
            assert conv4_3.get_shape().as_list()[1:] == [28, 28, 512]
            assert pool4.get_shape().as_list()[1:] == [14, 14, 512]

        # Fifth convolutional layer
        conv5_1 = self.conv_layer(pool4, "conv5_1", is_training)
        conv5_2 = self.conv_layer(conv5_1, "conv5_2", is_training)
        conv5_3 = self.conv_layer(conv5_2, "conv5_3", is_training)

        if debug:
            assert conv5_1.get_shape().as_list()[1:] == [14, 14, 512]
            assert conv5_2.get_shape().as_list()[1:] == [14, 14, 512]
            assert conv5_3.get_shape().as_list()[1:] == [14, 14, 512]

        # Backward upscale layer 5 and convolutional layers 4
        b_conv5_upscale = tf.image.resize_images(conv5_3, [28, 28], method=training_resize_method)
        b_conv4_input = tf.add(conv4_3, b_conv5_upscale, name="b_conv4_input")
        b_conv4 = self.conv_layer(b_conv4_input, "b_conv4", is_training)

        if debug:
            assert b_conv5_upscale.get_shape().as_list()[1:] == [28, 28, 512]
            assert b_conv4_input.get_shape().as_list()[1:] == [28, 28, 512]
            assert b_conv4.get_shape().as_list()[1:] == [28, 28, 256]

        # Backward upscale layer 4 and convolutional layers 3
        b_conv4_upscale = tf.image.resize_images(b_conv4, [56, 56], method=training_resize_method)
        b_conv3_input = tf.add(conv3_3, b_conv4_upscale, name="b_conv3_input")
        b_conv3 = self.conv_layer(b_conv3_input, "b_conv3", is_training)

        if debug:
            assert b_conv4_upscale.get_shape().as_list()[1:] == [56, 56, 256]
            assert b_conv3_input.get_shape().as_list()[1:] == [56, 56, 256]
            assert b_conv3.get_shape().as_list()[1:] == [56, 56, 128]

        # Backward upscale layer 3 and convolutional layers 2
        b_conv3_upscale = tf.image.resize_images(b_conv3, [112, 112], method=training_resize_method)
        b_conv2_input = tf.add(conv2_2, b_conv3_upscale, name="b_conv2_input")
        b_conv2 = self.conv_layer(b_conv2_input, "b_conv2", is_training)

        if debug:
            assert b_conv3_upscale.get_shape().as_list()[1:] == [112, 112, 128]
            assert b_conv2_input.get_shape().as_list()[1:] == [112, 112, 128]
            assert b_conv2.get_shape().as_list()[1:] == [112, 112, 64]

        # Backward upscale layer 2 and convolutional layers 1
        b_conv2_upscale = tf.image.resize_images(b_conv2, [224, 224], method=training_resize_method)
        b_conv1_input = tf.add(conv1_2, b_conv2_upscale, name="b_conv1_input")
        b_conv1 = self.conv_layer(b_conv1_input, "b_conv1", is_training)

        if debug:
            assert b_conv2_upscale.get_shape().as_list()[1:] == [224, 224, 64]
            assert b_conv1_input.get_shape().as_list()[1:] == [224, 224, 64]
            assert b_conv1.get_shape().as_list()[1:] == [224, 224, 3]

        # Output layer
        b_conv0 = self.conv_layer(tf.add(input_data, b_conv1), "b_conv0", is_training)

        if debug:
            assert b_conv0.get_shape().as_list()[1:] == [224, 224, 3]

        output_layer = tf.tanh(tf.slice(b_conv0, [0, 0, 0, 1], [-1, -1, -1, 2]), name='output_layer')

        return output_layer
