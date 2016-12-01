"""
Test model
"""

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
    test_file_paths = init_file_path(test_dir)

    # Init placeholder and global step
    print "Init placeholder"
    is_training = tf.placeholder(tf.bool, name="training_flag")
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Init vgg16 model
    print "Init vgg16 model"
    vgg = vgg16.Vgg16()

    # Init residual encoder model
    print "Init residual encoder model"
    residual_encoder = ResidualEncoder()

    color_image_rgb = input_pipeline(test_file_paths, 1)
    color_image_yuv = rgb_to_yuv(color_image_rgb)

    gray_image = tf.image.rgb_to_grayscale(color_image_rgb, name="gray_image")
    gray_image_rgb = tf.image.grayscale_to_rgb(gray_image, name="gray_image_rgb")
    gray_image_yuv = rgb_to_yuv(gray_image_rgb)
    gray_image_norm = tf.mul(tf.sub(gray_image, y_norm_para), 2.0, name="gray_image_norm")
    gray_image_input = tf.concat(concat_dim=3,
                                 values=[gray_image_norm, gray_image_norm, gray_image_norm],
                                 name="gray_image_input")

    with tf.name_scope("content_vgg"):
        vgg.build(gray_image_input)

    predict = residual_encoder.build(input_data=gray_image_input, vgg=vgg, is_training=is_training)
    cost = residual_encoder.get_cost(predict_val=predict, real_val=tf.slice(color_image_yuv, [0, 0, 0, 1], [-1, -1, -1, 2], name="color_image_uv"))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) \
        .minimize(cost, global_step=global_step, gate_gradients=tf.train.GradientDescentOptimizer.GATE_NONE)

    predict_yuv = tf.concat(concat_dim=3, values=[tf.slice(gray_image_yuv, [0, 0, 0, 0], [-1, -1, -1, 1], name="gray_image_y"), predict], name="predict_yuv")
    predict_rgb = yuv_to_rgb(predict_yuv)

    # Summaries
    print "Init summaries"
    tf.histogram_summary("output_conv", weights["output_conv"])
    tf.histogram_summary("b_conv4", weights["b_conv4"])
    tf.histogram_summary("b_conv3", weights["b_conv3"])
    tf.histogram_summary("b_conv2", weights["b_conv2"])
    tf.histogram_summary("b_conv1", weights["b_conv1"])
    tf.histogram_summary("b_conv0", weights["b_conv0"])
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
        test_writer = tf.train.SummaryWriter("summary/test")

        # Start input enqueue threads.
        print "Start input enqueue threads"
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Load model
        ckpt = tf.train.get_checkpoint_state("summary")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Load model finished!"
        else:
            print "Failed to restore model"
            exit()

        # Start testing
        print "Start testing!!!"

        # Init debug value
        max_y, min_y = -1, 1
        max_u, min_u = -1, 1
        max_v, min_v = -1, 1
        pred_max_u, pred_min_u = -1, 1
        pred_max_v, pred_min_v = -1, 1

        try:
            step = 0
            while not coord.should_stop():
                step += 1

                # Print batch loss
                if step % display_step == 0:
                    loss, pred, pred_rgb, color_rgb, gray_rgb, gray, color_yuv, summary = \
                        sess.run([cost, predict, predict_rgb, color_image_rgb, gray_image_rgb, gray_image_input,
                                  color_image_yuv, merged], feed_dict={is_training: False})
                    print "Iter %d, Minibatch Loss = %f" % (step, np.mean(loss))
                    test_writer.add_summary(summary, step)
                    test_writer.flush()

                    batch_y = tf.slice(gray, [0, 0, 0, 0], [-1, -1, -1, 1], name="batch_y")
                    batch_u = tf.slice(color_yuv, [0, 0, 0, 1], [-1, -1, -1, 1], name="batch_u")
                    batch_v = tf.slice(color_yuv, [0, 0, 0, 2], [-1, -1, -1, 1], name="batch_v")
                    max_y = np.maximum(sess.run(tf.reduce_max(batch_y)), max_y)
                    min_y = np.minimum(sess.run(tf.reduce_min(batch_y)), min_y)
                    max_u = np.maximum(sess.run(tf.reduce_max(batch_u)), max_u)
                    min_u = np.minimum(sess.run(tf.reduce_min(batch_u)), min_u)
                    max_v = np.maximum(sess.run(tf.reduce_max(batch_v)), max_v)
                    min_v = np.minimum(sess.run(tf.reduce_min(batch_v)), min_v)

                    # Save output image
                    zero_x = tf.zeros(tf.shape(batch_y))
                    pred_u = tf.slice(pred, [0, 0, 0, 0], [-1, -1, -1, 1], name="pred_u")
                    pred_v = tf.slice(pred, [0, 0, 0, 1], [-1, -1, -1, 1], name="pred_v")
                    u_image = yuv_to_rgb(tf.concat(concat_dim=3, values=[zero_x, pred_u, zero_x]))
                    v_image = yuv_to_rgb(tf.concat(concat_dim=3, values=[zero_x, zero_x, pred_v]))
                    uu_image = yuv_to_rgb(tf.concat(concat_dim=3, values=[zero_x, batch_u, zero_x]))
                    vv_image = yuv_to_rgb(tf.concat(concat_dim=3, values=[zero_x, zero_x, batch_v]))
                    summary_image = concat_images(gray_rgb[0], u_image[0])
                    summary_image = concat_images(summary_image, v_image[0])
                    summary_image = concat_images(summary_image, pred_rgb[0])
                    summary_image = concat_images(summary_image, uu_image[0])
                    summary_image = concat_images(summary_image, vv_image[0])
                    summary_image = concat_images(summary_image, color_rgb[0])
                    plt.imsave("summary/result/" + str(step) + "_0.jpg", sess.run(summary_image))

                    pred_max_u = np.maximum(sess.run(tf.reduce_max(pred_u)), pred_max_u)
                    pred_min_u = np.minimum(sess.run(tf.reduce_min(pred_u)), pred_min_u)
                    pred_max_v = np.maximum(sess.run(tf.reduce_max(pred_v)), pred_max_v)
                    pred_min_v = np.minimum(sess.run(tf.reduce_min(pred_v)), pred_min_v)

                if step == test_iters:
                    break

        except tf.errors.OUT_OF_RANGE as e:
            # Handle exception
            print "Done training -- epoch limit reached"
            coord.request_stop(e)

        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        print "Y in %f - %f" % (min_y, max_y)
        print "U in %f - %f" % (min_u, max_u)
        print "V in %f - %f" % (min_v, max_v)
        print "Pred_U in %f - %f" % (pred_min_u, pred_max_u)
        print "Pred_V in %f - %f" % (pred_min_v, pred_max_v)

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
