"""
Training model
"""

from config import *
from image_helper import (rgb_to_yuv, yuv_to_rgb)
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

    # Init vgg16 model
    print "Init vgg16 model"
    vgg = vgg16.Vgg16()

    # Init residual encoder model
    print "Init residual encoder model"
    residual_encoder = ResidualEncoder()

    # Create predict, cost and training model
    print "Create predict, cost and training model"
    color_image_rgb = input_pipeline(train_file_paths, batch_size)
    color_image_yuv = rgb_to_yuv(color_image_rgb)

    gray_image = tf.image.rgb_to_grayscale(color_image_rgb, name="gray_image")
    gray_image_rgb = tf.image.grayscale_to_rgb(gray_image, name="gray_image_rgb")
    gray_image_yuv = rgb_to_yuv(gray_image_rgb)
    gray_image = tf.mul(tf.sub(gray_image, y_norm_para), 2.0, name="gray_image_norm")  # Normalize input to -1..1
    gray_image = tf.concat(concat_dim=3, values=[gray_image, gray_image, gray_image], name="gray_image_input")

    with tf.name_scope("content_vgg"):
        vgg.build(gray_image)

    predict = residual_encoder.build(input_data=gray_image, vgg=vgg, is_training=is_training)
    cost = residual_encoder.get_cost(predict_val=predict, real_val=tf.slice(color_image_yuv, [0, 0, 0, 1], [-1, -1, -1, 2], name="color_image_uv"))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

    predict_yuv = tf.concat(concat_dim=3, values=[tf.slice(gray_image_yuv, [0, 0, 0, 0], [-1, -1, -1, 1], name="gray_image_y"), predict], name="predict_yuv")
    predict_rgb = yuv_to_rgb(predict_yuv)

    # Summaries
    print "Init summaries"
    tf.histogram_summary("cost", cost)
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
                step = sess.run(global_step)

                # Run optimizer
                sess.run(optimizer, feed_dict={is_training: True})

                # Print batch loss
                if step % display_step == 0:
                    loss, summary = sess.run([cost, merged], feed_dict={is_training: True})
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
