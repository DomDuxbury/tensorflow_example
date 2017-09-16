import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(layer_input, w_shape, b_shape):
    with tf.name_scope("conv"):
        W_conv = weight_variable(w_shape)
        b_conv = bias_variable(b_shape)
        h_conv = tf.nn.relu(conv2d(layer_input, W_conv) + b_conv)
        h_pool = max_pool_2x2(h_conv)
        return h_pool


def dense_layer(layer_input):
    with tf.name_scope("dense"):
        W_fc = weight_variable([7 * 7 * 64, 1024])
        b_fc = bias_variable([1024])

        h_pool2_flat = tf.reshape(layer_input, [-1, 7*7*64])
        h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
        return h_fc


def output_layer(layer_input):
    with tf.name_scope("output"):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        predictions = tf.matmul(layer_input, W_fc2) + b_fc2
        return predictions
