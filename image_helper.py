"""
Helper functions for image manipulation
"""

from config import *


def rgb_to_yuv(rgb_image):
    """
    Convert image color space from RGB to YUV
    :param rgb_image: an image with RGB color space
    :return: an image with YUV color space
    """
    # Get r, g, b channel
    _r = tf.slice(rgb_image, [0, 0, 0, 0], [-1, -1, -1, 1])
    _g = tf.slice(rgb_image, [0, 0, 0, 1], [-1, -1, -1, 1])
    _b = tf.slice(rgb_image, [0, 0, 0, 2], [-1, -1, -1, 1])

    # Calculate y, u, v channel
    _y = (0.299 * _r) + (0.587 * _g) + (0.114 * _b)
    _u = (-0.14713 * _r) - (0.28886 * _g) + (0.436 * _b)
    _v = (0.615 * _r) - (0.51499 * _g) - (0.10001 * _b)

    # Get image with YUV color space
    _yuv = tf.concat(concat_dim=3, values=[_y, _u, _v])

    # Normalize y, u, v channels
    _yuv = normalized_yuv(_yuv)

    return _yuv


def yuv_to_rgb(yuv_image):
    """
    Convert image color space from YUV to RGB
    :param yuv_image: an image with YUV color space
    :return: an image with RGB color space
    """
    # Denormalize y, u, v channels
    yuv_image = denormalized_yuv(yuv_image)

    # Get y, u, v channel
    _y = tf.slice(yuv_image, [0, 0, 0, 0], [-1, -1, -1, 1])
    _u = tf.slice(yuv_image, [0, 0, 0, 1], [-1, -1, -1, 1])
    _v = tf.slice(yuv_image, [0, 0, 0, 2], [-1, -1, -1, 1])

    # Calculate r, g, b channel
    _r = _y + 1.13983 * _v
    _g = _y - 0.39464 * _u - 0.58060 * _v
    _b = _y + 2.03211 * _u

    # Get image with RGB color space
    _rgb = tf.concat(concat_dim=3, values=[_r, _g, _b])

    return _rgb


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
    return tf.concat(concat_dim=1, values=[img_a, img_b])
