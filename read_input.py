"""
Helper functions for read input
"""

import os

from config import *
from image_helper import rgb_to_yuv


def init_file_path(directory):
    """
    Get the image file path array
    :param directory: the directory that store images
    :return: an array of image file path
    """
    paths = []
    for file_name in os.listdir(directory):
        # Skip files that is not jpg
        if not file_name.endswith('.jpg'):
            continue
        file_path = '%s/%s' % (directory, file_name)
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
    channel_y = tf.mul(tf.sub(channel_y, y_norm_para), 2.0, name="channel_y")
    channel_u = tf.div(channel_u, u_norm_para, name="channel_u")
    channel_v = tf.div(channel_v, v_norm_para, name="channel_v")
    # Add channel data
    channel_uv = tf.concat(concat_dim=3, values=[channel_u, channel_v], name="channel_uv")
    return channel_y, channel_uv
