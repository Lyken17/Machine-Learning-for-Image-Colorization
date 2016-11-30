from time import sleep

from image_helper import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

img = mpimg.imread("train2014/250_250.jpg")
imgplot = plt.imshow(img)
print img.shape
plt.show()

# Init the graph
print "Init graph"
init = tf.initialize_all_variables()

sess = tf.Session()
# Initialize the variables.
sess.run(init)

# Make pixel element value in [0, 1)
img = tf.image.convert_image_dtype(img, tf.float32, name="float_image")
img = tf.image.resize_images(img, [224, 224])
img = tf.pack([img])
print sess.run(tf.shape(img))

img_yuv = rgb_to_yuv(img)
y = tf.slice(img_yuv[0], [0, 0, 0], [-1, -1, 1])
u = tf.slice(img_yuv[0], [0, 0, 1], [-1, -1, 1])
v = tf.slice(img_yuv[0], [0, 0, 2], [-1, -1, 1])
print sess.run(tf.reduce_min(y)), sess.run(tf.reduce_max(y))
print sess.run(tf.reduce_min(u)), sess.run(tf.reduce_max(u))
print sess.run(tf.reduce_min(v)), sess.run(tf.reduce_max(v))

gray_image = tf.image.rgb_to_grayscale(img, name="gray_image")
gray_image_rgb = tf.image.grayscale_to_rgb(gray_image, name="gray_image_rgb")
gray_image_yuv = rgb_to_yuv(gray_image_rgb)
gray_image = tf.mul(tf.sub(gray_image, y_norm_para), 2.0, name="gray_image_norm")  # Normalize input to -1..1
gray_image = tf.concat(concat_dim=3, values=[gray_image, gray_image, gray_image], name="gray_image_input")

imgplot = plt.imshow(sess.run(gray_image_rgb[0]))
plt.show()

img_rgb = yuv_to_rgb(img_yuv)
imgplot = plt.imshow(sess.run(img_rgb[0]))
plt.show()

sess.close()
