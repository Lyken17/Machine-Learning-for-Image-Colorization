"""
Helper functions for image manipulation
"""

import numpy as np
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
    _y = (0.299 * _r) + (0.587 * _g) + (0.114 * _b)
    _u = (-0.14713 * _r) - (0.28886 * _g) + (0.436 * _b)
    _v = (0.615 * _r) - (0.51499 * _g) - (0.10001 * _b)

    # Get image with YUV color space
    _yuv = tf.concat(concat_dim=2, values=[_y, _u, _v])

    return _yuv


def yuv_to_rgb(yuv_image):
    """
    Convert image color space from YUV to RGB
    :param yuv_image: an image with YUV color space
    :return: an image with RGB color space
    """
    # Get width and height for image
    _w = tf.shape(yuv_image)[0]
    _h = tf.shape(yuv_image)[1]

    # Get y, u, v channel
    _y = tf.slice(yuv_image, [0, 0, 0], [_w, _h, 1])
    _u = tf.slice(yuv_image, [0, 0, 1], [_w, _h, 1])
    _v = tf.slice(yuv_image, [0, 0, 2], [_w, _h, 1])

    # Calculate r, g, b channel
    _r = _y + 1.13983 * _v
    _g = _y - 0.39464 * _u - 0.58060 * _v
    _b = _y + 2.03211 * _u

    # Get image with RGB color space
    _rgb = tf.concat(concat_dim=2, values=[_r, _g, _b])
    _rgb = tf.image.convert_image_dtype(_rgb, tf.uint8, saturate=True)

    return _rgb


def concat_images(img_a, img_b):
    """
    Combines two color image side-by-side.
    :param img_a: image a on left
    :param img_b: image b on right
    :return: combined image
    """
    new_img = tf.concat(concat_dim=0, values=[img_a, img_b])
    return new_img
