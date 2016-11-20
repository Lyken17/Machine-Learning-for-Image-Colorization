"""
Residual-encoder model implementation.

See extensive documentation at
http://tinyclouds.org/colorize/
"""

import os

import numpy as np
from tensorflow.contrib.layers import batch_norm
from matplotlib import pyplot as plt

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


def read_image(filename_queue):
    """
    Read and store image with YUV color space
    :param filename_queue: the filename queue for image files
    :return: image with YUV color space
    """
    # Read the image with RGB color space
    reader = tf.WholeFileReader()
    key, content = reader.read(filename_queue)
    rgb_image = tf.image.decode_jpeg(content, channels=3)
    # Make pixel element value in [0, 1)
    rgb_image = tf.image.convert_image_dtype(rgb_image, tf.float32)
    # Resize image to the right image_size
    rgb_image = tf.image.resize_images(rgb_image, [image_size, image_size], method=input_resize_method)
    # Change color space to YUV
    yuv_image = rgb_to_yuv(rgb_image)
    return yuv_image


def input_pipeline(filenames, b_size, num_epochs=None, shuffle=True):
    """
    Use a queue that randomizes the order of examples and return batch of images
    :param filenames: filenames
    :param b_size: batch size
    :param num_epochs: number of epochs for producing each string before generating an OutOfRange error
    :param shuffle: if true, the strings are randomly shuffled within each epoch
    :return: a batch of yuv_images
    """
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
    yuv_image = read_image(filename_queue)
    min_after_dequeue = dequeue_buffer_size
    capacity = min_after_dequeue + 3 * b_size
    image_batch = tf.train.shuffle_batch([yuv_image],
                                         batch_size=b_size,
                                         capacity=capacity,
                                         shapes=[image_size, image_size, 3],
                                         min_after_dequeue=min_after_dequeue)
    return image_batch


def get_y_and_uv_data(filenames, b_size):
    """
    Get the input data with Y channel and target data with UV channels
    :param filenames: the path for the image file
    :param b_size: batch size
    :return: input data with Y channel and target data with UV channels
    """
    # Get the image with YUV channels
    _yuv = input_pipeline(filenames, b_size)
    # Split channels
    channel_y = tf.slice(_yuv, [0, 0, 0, 0], [-1, -1, -1, 1])
    channel_u = tf.slice(_yuv, [0, 0, 0, 1], [-1, -1, -1, 1])
    channel_v = tf.slice(_yuv, [0, 0, 0, 2], [-1, -1, -1, 1])
    # Normalize channels
    channel_y = tf.mul(tf.sub(channel_y, 0.5), 2.0, name="channel_y")
    channel_u = tf.div(channel_u, 0.436, name="channel_u")
    channel_v = tf.div(channel_v, 0.615, name="channel_v")
    # Add channel data
    channel_uv = tf.concat(concat_dim=3, values=[channel_u, channel_v], name="channel_uv")
    return channel_y, channel_uv


def concat_images(img_a, img_b):
    """
    Combines two color image side-by-side.
    :param img_a: image a on left
    :param img_b: image b on right
    :return: combined image
    """
    height_a, width_a = img_a.shape[:2]
    height_b, width_b = img_b.shape[:2]
    max_height = np.max([height_a, height_b])
    total_width = width_a + width_b
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:height_a, :width_a] = img_a
    new_img[:height_b, width_a:total_width] = img_b
    return new_img


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

    # @staticmethod
    # def get_bias(name):
    #     """
    #     Get bias for one layer
    #     :param name: the name of the layer
    #     :return: the initial bias for this layer
    #     """
    #     return biases[name]

    @staticmethod
    def get_cost(predict_val, real_val):
        """
        Cost function
        :param predict_val: the predict value
        :param real_val: the real value
        :return: cost
        """
        return tf.reduce_mean(tf.square(tf.sub(predict_val, real_val)))

    @staticmethod
    def get_accuracy(predict_val, real_val):
        """
        Accuracy function
        :param predict_val: the predict value
        :param real_val: the real value
        :return: accuracy
        """
        # TODO: Implement accuracy function
        return tf.reduce_mean(tf.square(tf.sub(predict_val, real_val)))

    def conv_layer(self, layer_input, name):
        """
        Convolution layer
        :param layer_input: the input data for this layer
        :param name: name for this layer
        :return: the layer data after convolution
        """
        with tf.variable_scope(name):
            weight = self.get_weight(name)
            output = tf.nn.conv2d(layer_input, weight, strides=[1, 1, 1, 1], padding='SAME')
            # conv_biases = self.get_bias(name)
            # output = tf.nn.bias_add(output, conv_biases)
            # output = tf.nn.relu(output)
            output = self.batch_normal(output, training_flag=is_training, scope=name)
            return output

    def max_pool(self, layer_input, name):
        """
        Polling layer
        :param layer_input: the input data for this layer
        :param name: name for this layer
        :return: the layer data after max-pooling
        """
        return tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def batch_normal(self, input_data, training_flag, scope):
        return tf.cond(training_flag,
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
        input_data = tf.concat(concat_dim=3, values=[input_data, input_data, input_data])

        if debug:
            assert input_data.get_shape().as_list()[1:] == [224, 224, 3]

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

        # Backward upscale layer 5 and convolutional layers 4
        b_conv5_upscale = tf.image.resize_images(conv5_3, [28, 28], method=training_resize_method)
        b_conv4 = self.conv_layer(tf.add(conv4_3, b_conv5_upscale), "b_conv4")

        if debug:
            assert b_conv5_upscale.get_shape().as_list()[1:] == [28, 28, 512]
            assert b_conv4.get_shape().as_list()[1:] == [28, 28, 256]

        # Backward upscale layer 4 and convolutional layers 3
        b_conv4_upscale = tf.image.resize_images(b_conv4, [56, 56], method=training_resize_method)
        b_conv3 = self.conv_layer(tf.add(conv3_3, b_conv4_upscale), "b_conv3")

        if debug:
            assert b_conv4_upscale.get_shape().as_list()[1:] == [56, 56, 256]
            assert b_conv3.get_shape().as_list()[1:] == [56, 56, 128]

        # Backward upscale layer 3 and convolutional layers 2
        b_conv3_upscale = tf.image.resize_images(b_conv3, [112, 112], method=training_resize_method)
        b_conv2 = self.conv_layer(tf.add(conv2_2, b_conv3_upscale), "b_conv2")

        if debug:
            assert b_conv3_upscale.get_shape().as_list()[1:] == [112, 112, 128]
            assert b_conv2.get_shape().as_list()[1:] == [112, 112, 64]

        # Backward upscale layer 2 and convolutional layers 1
        b_conv2_upscale = tf.image.resize_images(b_conv2, [224, 224], method=training_resize_method)
        b_conv1 = self.conv_layer(tf.add(conv1_2, b_conv2_upscale), "b_conv1")

        if debug:
            assert b_conv2_upscale.get_shape().as_list()[1:] == [224, 224, 64]
            assert b_conv1.get_shape().as_list()[1:] == [224, 224, 3]

        # Output layer
        b_conv0 = self.conv_layer(tf.add(input_data, b_conv1), "b_conv0")

        if debug:
            assert b_conv0.get_shape().as_list()[1:] == [224, 224, 3]

        output_layer = tf.tanh(tf.slice(b_conv0, [0, 0, 0, 1], [-1, -1, -1, 2]), name='output_layer')

        return output_layer


if __name__ == '__main__':
    # Init image data file path
    train_file_paths = init_file_path(train_dir)
    test_file_paths = init_file_path(test_dir)

    # Init placeholder and global step
    is_training = tf.placeholder(tf.bool, name="training_flag")
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Init residual encoder model
    residual_encoder = ResidualEncoder()

    # Make predict, cost and training model
    batch_xs, batch_ys = get_y_and_uv_data(train_file_paths, batch_size)
    predict = residual_encoder.build(input_data=batch_xs)
    cost = residual_encoder.get_cost(predict_val=predict, real_val=batch_ys)
    accuracy = residual_encoder.get_accuracy(predict_val=predict, real_val=batch_ys)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

    # Summaries
    tf.histogram_summary("conv1_1", weights["conv1_1"])
    tf.histogram_summary("conv1_2", weights["conv1_2"])
    tf.histogram_summary("conv2_1", weights["conv2_1"])
    tf.histogram_summary("conv2_2", weights["conv2_2"])
    tf.histogram_summary("conv3_1", weights["conv3_1"])
    tf.histogram_summary("conv3_2", weights["conv3_2"])
    tf.histogram_summary("conv3_3", weights["conv3_3"])
    tf.histogram_summary("conv4_1", weights["conv4_1"])
    tf.histogram_summary("conv4_2", weights["conv4_2"])
    tf.histogram_summary("conv4_3", weights["conv4_3"])
    tf.histogram_summary("conv5_1", weights["conv5_1"])
    tf.histogram_summary("conv5_2", weights["conv5_2"])
    tf.histogram_summary("conv5_3", weights["conv5_3"])
    tf.histogram_summary("b_conv4", weights["b_conv4"])
    tf.histogram_summary("b_conv3", weights["b_conv3"])
    tf.histogram_summary("b_conv2", weights["b_conv2"])
    tf.histogram_summary("b_conv1", weights["b_conv1"])
    tf.histogram_summary("b_conv0", weights["b_conv0"])
    tf.histogram_summary("cost", cost)
    # tf.image_summary("colorimage", colorimage, max_images=1)
    # tf.image_summary("pred_rgb", pred_rgb, max_images=1)
    # tf.image_summary("grayscale", grayscale_rgb, max_images=1)

    # Saver
    saver = tf.train.Saver()

    # Create the graph, etc
    init = tf.initialize_all_variables()

    # Create a session for running operations in the Graph
    with tf.Session() as sess:
        # Initialize the variables.
        sess.run(init)

        # Merge all summaries
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter("summary/train", sess.graph)
        test_writer = tf.train.SummaryWriter("summary/test")

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Start training
        print "Start training!!!"

        try:
            while not coord.should_stop():
                step = sess.run(global_step)
                if step == training_iters:
                    break
                sess.run(optimizer, feed_dict={is_training: True})
                if step % display_step == 0:
                    loss, acc, summary = sess.run([cost, accuracy, merged], feed_dict={is_training: True})
                    print "Iter %d, Minibatch Loss = %f, Training Accuracy = %f" % \
                          (step, loss, acc)
                #    summary_image = concat_images(gray_image[0], pred_image[0])
                #    summary_image = concat_images(summary_image, color_image[0])
                #    plt.imsave("summary/result/" + str(step) + "_0", summary_image)
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

            print "Training Finished!"
            # Predict
            # TODO: Test with testing data
            # print "Testing Accuracy: %f" % (sess.run(accuracy, feed_dict={x: 0, y: 0, is_training: False}))
        except tf.errors.OUT_OF_RANGE as e:
            # Handle exception
            print('Done training -- epoch limit reached')
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
