"""
Training model
"""

import numpy as np

from config import *
from read_input import (init_file_path, get_y_and_uv_data)
from residual_encoder import ResidualEncoder


if __name__ == '__main__':
    # Init image data file path
    print "Init file path"
    train_file_paths = init_file_path(train_dir)

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
    batch_xs, batch_ys = get_y_and_uv_data(train_file_paths, batch_size)
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
        train_writer = tf.train.SummaryWriter("summary/train", sess.graph)

        # Start input enqueue threads.
        print "Start input enqueue threads"
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Start training
        print "Start training!!!"

        try:
            while not coord.should_stop():
                step = sess.run(global_step)

                # Get batch of data
                batch_x, batch_y = sess.run([batch_xs, batch_ys])

                # Run optimizer
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, is_training: True})

                # Print batch loss
                if step % display_step == 0:
                    loss, summary = sess.run([cost, merged], feed_dict={x: batch_x, y: batch_y, is_training: True})
                    print "Iter %d, Minibatch Loss = %f" % (step, loss)
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                # Save model
                if step % save_step == 0 and step != 0:
                    save_path = saver.save(sess, "summary/model.ckpt")
                    print "Model saved in file: %s" % save_path

                # Stop training
                if step == training_iters:
                    break

            print "Training Finished!"

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
