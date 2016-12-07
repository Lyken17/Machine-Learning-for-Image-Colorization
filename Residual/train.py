"""
Training model
"""

import sys

import numpy as np
from matplotlib import pyplot as plt

from config import *
from image_helper import (rgb_to_yuv, yuv_to_rgb, concat_images)
from read_input import (init_file_path, input_pipeline)
from residual_encoder import ResidualEncoder
from vgg import vgg16

if __name__ == '__main__':
    # Init image data file path
    print "Init file path"
    train_file_paths = init_file_path(train_dir)

    # Init placeholder and global step
    print "Init placeholder"
    is_training = tf.placeholder(tf.bool, name="training_flag")
    global_step = tf.Variable(0, name='global_step', trainable=False)
    uv = tf.placeholder(tf.uint8, name='uv')

    # Init vgg16 model
    print "Init vgg16 model"
    vgg = vgg16.Vgg16()

    # Init residual encoder model
    print "Init residual encoder model"
    residual_encoder = ResidualEncoder()

    # Color image
    color_image_rgb = input_pipeline(train_file_paths, batch_size)
    color_image_yuv = rgb_to_yuv(color_image_rgb)

    # Gray image
    gray_image = tf.image.rgb_to_grayscale(color_image_rgb, name="gray_image")
    gray_image_rgb = tf.image.grayscale_to_rgb(gray_image, name="gray_image_rgb")
    gray_image_yuv = rgb_to_yuv(gray_image_rgb)
    gray_image = tf.concat(concat_dim=3, values=[gray_image, gray_image, gray_image])

    # Build vgg model
    with tf.name_scope("content_vgg"):
        vgg.build(gray_image)

    # Predict model
    predict = residual_encoder.build(input_data=gray_image, vgg=vgg, is_training=is_training)
    predict_yuv = tf.concat(concat_dim=3, values=[tf.slice(gray_image_yuv, [0, 0, 0, 0], [-1, -1, -1, 1], name="gray_image_y"), predict], name="predict_yuv")
    predict_rgb = yuv_to_rgb(predict_yuv)

    # Cost
    cost = residual_encoder.get_cost(predict_val=predict, real_val=tf.slice(color_image_yuv, [0, 0, 0, 1], [-1, -1, -1, 2], name="color_image_uv"))

    if uv == 1:
        cost = tf.split(3, 2, cost)[0]
    elif uv == 2:
        cost = tf.split(3, 2, cost)[1]
    else:
        cost = (tf.split(3, 2, cost)[0] + tf.split(3, 2, cost)[1]) / 2

    # Optimizer
    if is_training is not None:
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = opt.minimize(cost, global_step=global_step, gate_gradients=opt.GATE_NONE)

    # Summaries
    print "Init summaries"
    tf.histogram_summary("cost", tf.reduce_mean(cost))
    tf.image_summary("color_image_rgb", color_image_rgb, max_images=1)
    tf.image_summary("predict_rgb", predict_rgb, max_images=1)
    tf.image_summary("gray_image", gray_image_rgb, max_images=1)

    # Saver
    print "Init model saver"
    saver = tf.train.Saver()

    # Init the graph
    print "Init graph"
    init = tf.initialize_all_variables()

    # Create a session for running operations in the Graph
    with tf.Session() as sess:
        # Initialize the variables.
        sess.run(init)

        # Merge all summaries
        print "Merge all summaries"
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter("summary/train", sess.graph)

        # Start input enqueue threads.
        print "Start input enqueue threads"
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Start training
        print "Start training!!!"

        try:
            while not coord.should_stop():
                # Run optimizer
                sess.run(optimizer, feed_dict={is_training: True, uv: 1})
                sess.run(optimizer, feed_dict={is_training: True, uv: 2})
                step = sess.run(global_step)

                # Print batch loss
                if step % display_step == 0:
                    loss, pred, color, gray, summary = sess.run([cost, predict_rgb, color_image_rgb, gray_image_rgb, merged],
                                                                feed_dict={is_training: False, uv: 3})
                    print "Iter %d, Minibatch Loss = %f" % (step, np.mean(loss))
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                    # Save test image
                    if step % test_step == 0:
                        summary_image = concat_images(gray[0], pred[0])
                        summary_image = concat_images(summary_image, color[0])
                        plt.imsave("summary/result/" + str(step) + "_0.jpg", summary_image)

                # Save model
                if step % save_step == 0 and step != 0:
                    save_path = saver.save(sess, "summary/model.ckpt")
                    print "Model saved in file: %s" % save_path

                # Stop training
                if step == training_iters:
                    break

            print "Training Finished!"
            sys.stdout.flush()

        except tf.errors.OUT_OF_RANGE as e:
            # Handle exception
            print "Done training -- epoch limit reached"
            coord.request_stop(e)

        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
