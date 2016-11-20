"""
Download images from placekitten.com
This is only used for funny test
"""

from PIL import Image
from StringIO import StringIO

import requests


root_url = 'https://placekitten.com'
root_train_dir = 'train2014'
root_val_dir = 'val2014'


def download_images(url):
    """
    Download images from placekitten.com
    :param url: request url
    :return: None
    """
    # Download training images
    for w in range(250, 500, 5):
        for h in range(250, 500, 5):
            image_url = '%s/%s/%s' % (url, w, h)
            res = requests.get(image_url)
            image = Image.open(StringIO(res.content))
            image.save('%s/%s_%s.jpg' % (root_train_dir, w, h))

    # Download validation images
    for w in range(500, 600, 5):
        for h in range(500, 600, 5):
            image_url = '%s/%s/%s' % (url, w, h)
            res = requests.get(image_url)
            image = Image.open(StringIO(res.content))
            image.save('%s/%s_%s.jpg' % (root_val_dir, w, h))


if __name__ == '__main__':
    download_images(root_url)
