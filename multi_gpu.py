import time

import numpy as np
import tensorflow as tf
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True, seed=1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Multi-GPU CNN')
    parser.add_argument('--num_gpus', dest='num_gpus', help='number of GPUs',default=2,type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size',default=100,type=int)
    args = parser.parse_args()
    return args


def create_weight_and_bias(weight_shape, bias_shape):
    with tf.device('/cpu:0'):
        weight = tf.get_variable("weight", initializer=tf.truncated_normal(weight_shape, stddev=0.1,dtype=tf.float32))
        bias = tf.get_variable("bias", initializer= tf.constant(0, shape=bias_shape,dtype=tf.float32))
        return weight, bias


def create_conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def create_max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def create_average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def create_cnn(x_, y_, keep_prob, reuse=False):
    x_image = tf.reshape(x_, [-1, 28, 28, 1])

    with tf.variable_scope("conv1", reuse=reuse):
        W_conv1, b_conv1 = create_weight_and_bias([5, 5, 1, 32], [32])
        h_conv1 = tf.nn.relu(create_conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = create_max_pool_2x2(h_conv1)

    with tf.variable_scope("conv2", reuse=reuse):
        W_conv2, b_conv2 = create_weight_and_bias([5, 5, 32, 64], [64])
        h_conv2 = tf.nn.relu(create_conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = create_max_pool_2x2(h_conv2)

    with tf.variable_scope("fc1", reuse=reuse):
        W_fc1, b_fc1 = create_weight_and_bias([7 * 7 * 64, 1024], [1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.variable_scope("fc2", reuse=reuse):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        W_fc2, b_fc2 = create_weight_and_bias([1024, 10], [10])
        y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_mean(y_ * tf.log(y))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.cast(correct_prediction, tf.float32)

    return cross_entropy, accuracy

if __name__ == '__main__':
    args = parse_args()
    batch_size = args.batch_size
    num_gpus = args.num_gpus

    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):
        optimizer = tf.train.AdamOptimizer(1e-4)
        x_all = tf.placeholder(tf.float32, [batch_size * num_gpus, 784], name="x_all")
        y_all = tf.placeholder(tf.float32, [batch_size * num_gpus, 10], name="y_all")
        keep_prob = tf.placeholder(tf.float32)

        tower_grads = []
        tower_acc = []
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % (i)) as scope:
                    if i > 0:
                        reuse = True
                    else:
                        reuse = False
                    x_next = x_all[i * batch_size:(i + 1) * batch_size, :]
                    y_next = y_all[i * batch_size:(i + 1) * batch_size, :]
                    loss, acc = create_cnn(x_next, y_next, keep_prob, reuse=reuse)
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                    tower_acc.append(acc)

        avg_grads = create_average_gradients(tower_grads)
        avg_acc = tf.reduce_mean(tower_acc)
        trainer = optimizer.apply_gradients(avg_grads)

        init = tf.initialize_all_variables()

    with graph.as_default(), tf.device('/cpu:0'):
        time_start = time.time()
        saver = tf.train.Saver()
        sess = tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)

        for i in range(3000):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size * num_gpus)
            sess.run(trainer, feed_dict={x_all: batch_xs, y_all: batch_ys, keep_prob: 0.5})
            if i % 100 == 0:
                valid_accurarcy = 0
                for _ in range(mnist.validation.num_examples / (batch_size * num_gpus)):
                    batch_xs, batch_ys = mnist.validation.next_batch(batch_size * num_gpus)
                    valid_accurarcy += sess.run(avg_acc, feed_dict={x_all: batch_xs, y_all: batch_ys, keep_prob: 1})
                valid_accurarcy = valid_accurarcy / (mnist.validation.num_examples / (batch_size * num_gpus))
                print "i:%s, valid_accurarcy:%s" % (i, valid_accurarcy)

        valid_accurarcy = 0
        for _ in range(mnist.validation.num_examples / (batch_size * num_gpus)):
            batch_xs, batch_ys = mnist.validation.next_batch(batch_size * num_gpus)
            valid_accurarcy += sess.run(avg_acc, feed_dict={x_all: batch_xs, y_all: batch_ys, keep_prob: 1})
        valid_accurarcy = valid_accurarcy /  (mnist.validation.num_examples / (batch_size * num_gpus))
        print "valid_accurarcy:%s" % (valid_accurarcy)
        test_accurarcy = 0
        for _ in range(mnist.test.num_examples / (batch_size * num_gpus)):
            batch_xs, batch_ys = mnist.test.next_batch(batch_size * num_gpus)
            test_accurarcy += sess.run(avg_acc, feed_dict={x_all: batch_xs, y_all: batch_ys, keep_prob: 1})
        test_accurarcy = test_accurarcy /  (mnist.test.num_examples / (batch_size * num_gpus))
        print "test_accurarcy:%s" % (test_accurarcy)

        saver.save(sess, "model_multi_gpu.ckpt")
        time_end = time.time()
        print "total time:%s" % (time_end - time_start)
