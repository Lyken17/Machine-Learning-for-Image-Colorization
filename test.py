"""
Test model
"""

import numpy as np
from matplotlib import pyplot as plt

from config import *
from image_helper import (yuv_to_rgb, concat_images)
from read_input import (init_file_path, get_y_and_uv_data)
from residual_encoder import ResidualEncoder


if __name__ == '__main__':
    # Init image data file path
    print "Init file path"
    test_file_paths = init_file_path(test_dir)

    # Init placeholder and global step
    print "Init placeholder"
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 1], name="x")
    y = tf.placeholder(tf.float32, [None, image_size, image_size, 2], name="y")
    is_training = tf.placeholder(tf.bool, name="training_flag")
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Init residual encoder model
    print "Init residual encoder model"
    residual_encoder = ResidualEncoder()

    # Create predict, cost and training model
    print "Create predict, cost and training model"
    batch_xs, batch_ys = get_y_and_uv_data(test_file_paths, batch_size)
    predict = residual_encoder.build(input_data=x, is_training=is_training)
    cost = residual_encoder.get_cost(predict_val=predict, real_val=y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

    # Summaries
    print "Init summaries"
    tf.histogram_summary("cost", cost)

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
                if step == test_iters:
                    break

                # Get batch of data
                batch_x, batch_y = sess.run([batch_xs, batch_ys])

                _uu = tf.slice(batch_y, [0, 0, 0, 0], [-1, -1, -1, 1])
                _vv = tf.slice(batch_y, [0, 0, 0, 1], [-1, -1, -1, 1])
                max_y = np.maximum(sess.run(tf.reduce_max(batch_x)), max_y)
                min_y = np.minimum(sess.run(tf.reduce_min(batch_x)), min_y)
                max_u = np.maximum(sess.run(tf.reduce_max(_uu)), max_u)
                min_u = np.minimum(sess.run(tf.reduce_min(_uu)), min_u)
                max_v = np.maximum(sess.run(tf.reduce_max(_vv)), max_v)
                min_v = np.minimum(sess.run(tf.reduce_min(_vv)), min_v)

                # Print batch loss
                if step % display_step == 0:
                    loss, pred, summary = sess.run([cost, predict, merged], feed_dict={x: batch_x, y: batch_y, is_training: False})
                    print "Iter %d, Minibatch Loss = %f" % (step, loss)
                    test_writer.add_summary(summary, step)
                    test_writer.flush()

                    # Save output image
                    zero_x = tf.zeros(tf.shape(batch_x))
                    u_slice = tf.slice(pred, [0, 0, 0, 0], [-1, -1, -1, 1])
                    v_slice = tf.slice(pred, [0, 0, 0, 1], [-1, -1, -1, 1])
                    gray_image = yuv_to_rgb(tf.concat(concat_dim=3, values=[batch_x, zero_x, zero_x])[0])
                    u_image = yuv_to_rgb(tf.concat(concat_dim=3, values=[zero_x, u_slice, zero_x])[0])
                    v_image = yuv_to_rgb(tf.concat(concat_dim=3, values=[zero_x, zero_x, v_slice])[0])
                    pred_image = yuv_to_rgb(tf.concat(concat_dim=3, values=[batch_x, pred])[0])
                    color_image = yuv_to_rgb(tf.concat(concat_dim=3, values=[batch_x, batch_y])[0])
                    uu_image = yuv_to_rgb(tf.concat(concat_dim=3, values=[zero_x, _uu, zero_x])[0])
                    vv_image = yuv_to_rgb(tf.concat(concat_dim=3, values=[zero_x, zero_x, _vv])[0])
                    summary_image = concat_images(gray_image, u_image)
                    summary_image = concat_images(summary_image, v_image)
                    summary_image = concat_images(summary_image, pred_image)
                    summary_image = concat_images(summary_image, uu_image)
                    summary_image = concat_images(summary_image, vv_image)
                    summary_image = concat_images(summary_image, color_image)
                    plt.imsave("summary/result/" + str(step) + "_0.jpg", sess.run(summary_image))

                    pred_max_u = np.maximum(sess.run(tf.reduce_max(u_slice)), pred_max_u)
                    pred_min_u = np.minimum(sess.run(tf.reduce_min(u_slice)), pred_min_u)
                    pred_max_v = np.maximum(sess.run(tf.reduce_max(v_slice)), pred_max_v)
                    pred_min_v = np.minimum(sess.run(tf.reduce_min(v_slice)), pred_min_v)

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
