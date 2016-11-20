"""
Helper functions for color space conversion
"""

import tensorflow as tf


def rgb_to_yuv(rgb_image):
    """
    Convert image color space from RGB to YUV
    :param rgb_image: an image with RGB color space
    :return: an image with YUV color space
    """
    # Get width and height for image
    _w = tf.shape(rgb_image)[0]
    _h = tf.shape(rgb_image)[1]

    # Get r, g, b channel
    _r = tf.slice(rgb_image, [0, 0, 0], [_w, _h, 1])
    _g = tf.slice(rgb_image, [0, 0, 1], [_w, _h, 1])
    _b = tf.slice(rgb_image, [0, 0, 2], [_w, _h, 1])

    # Calculate y, u, v channel
    _y = (0.257 * _r) + (0.504 * _g) + (0.098 * _b) + 16
    _v = (0.439 * _r) - (0.368 * _g) - (0.071 * _b) + 128
    _u = -(0.148 * _r) - (0.291 * _g) + (0.439 * _b) + 128

    # Get image with yuv color space
    _yuv = tf.concat(concat_dim=2, values=[_y, _u, _v])

    return _yuv

