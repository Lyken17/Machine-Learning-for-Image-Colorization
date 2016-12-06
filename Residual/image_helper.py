"""
Helper functions for image manipulation
"""

import numpy as np

from config import *


def rgb_to_yuv(rgb_image):
    """
    Convert image color space from RGB to YUV
    :param rgb_image: an image with RGB color space
    :return: an image with YUV color space
    """
    rgb2yuv_filter = tf.constant(
        [[[[0.299, -0.169, 0.499],
           [0.587, -0.331, -0.418],
           [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0.0, 0.5, 0.5])

    yuv_image = tf.nn.conv2d(rgb_image, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    yuv_image = tf.nn.bias_add(yuv_image, rgb2yuv_bias)

    if normalize_yuv:
        # Normalize y, u, v channels
        yuv_image = normalized_yuv(yuv_image)

    return yuv_image


def yuv_to_rgb(yuv_image):
    """
    Convert image color space from YUV to RGB
    :param yuv_image: an image with YUV color space
    :return: an image with RGB color space
    """
    if normalize_yuv:
        # Denormalize y, u, v channels
        yuv_image = denormalized_yuv(yuv_image)

    yuv_image = tf.mul(yuv_image, 255)
    yuv2rgb_filter = tf.constant(
        [[[[1.0, 1.0, 1.0],
           [0.0, -0.34413999, 1.77199996],
           [1.40199995, -0.71414, 0.0]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])

    rgb_image = tf.nn.conv2d(yuv_image, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    rgb_image = tf.nn.bias_add(rgb_image, yuv2rgb_bias)
    rgb_image = tf.maximum(rgb_image, tf.zeros(rgb_image.get_shape(), dtype=tf.float32))
    rgb_image = tf.minimum(rgb_image, tf.mul(tf.ones(rgb_image.get_shape(), dtype=tf.float32), 255))
    rgb_image = tf.div(rgb_image, 255)

    return rgb_image


def normalized_yuv(yuv_images):
    """
    Normalize the yuv image data
    :param yuv_images: the YUV images that needs normalization
    :return: the normalized yuv image
    """
    # Split channels
    channel_y = tf.slice(yuv_images, [0, 0, 0, 0], [-1, -1, -1, 1])
    channel_u = tf.slice(yuv_images, [0, 0, 0, 1], [-1, -1, -1, 1])
    channel_v = tf.slice(yuv_images, [0, 0, 0, 2], [-1, -1, -1, 1])

    # Normalize y, u, v channels
    channel_y = tf.mul(tf.sub(channel_y, y_norm_para), 2.0, name="channel_y")
    channel_u = tf.div(channel_u, u_norm_para, name="channel_u")
    channel_v = tf.div(channel_v, v_norm_para, name="channel_v")

    # Add channel data
    channel_yuv = tf.concat(concat_dim=3, values=[channel_y, channel_u, channel_v], name="channel_yuv")
    return channel_yuv


def denormalized_yuv(yuv_images):
    """
    Denormalize the yuv image data
    :param yuv_images: the YUV images that needs denormalization
    :return: the denormalized yuv image
    """
    # Split channels
    channel_y = tf.slice(yuv_images, [0, 0, 0, 0], [-1, -1, -1, 1])
    channel_u = tf.slice(yuv_images, [0, 0, 0, 1], [-1, -1, -1, 1])
    channel_v = tf.slice(yuv_images, [0, 0, 0, 2], [-1, -1, -1, 1])

    # Denormalize y, u, v channels
    channel_y = tf.add(tf.div(channel_y, 2.0), y_norm_para, name="channel_y")
    channel_u = tf.mul(channel_u, u_norm_para, name="channel_u")
    channel_v = tf.mul(channel_v, v_norm_para, name="channel_v")

    # Add channel data
    channel_yuv = tf.concat(concat_dim=3, values=[channel_y, channel_u, channel_v], name="channel_yuv")
    return channel_yuv


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
