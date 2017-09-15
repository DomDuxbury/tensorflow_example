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
    W_conv = weight_variable(w_shape)
    b_conv = bias_variable(b_shape)
    h_conv = tf.nn.relu(conv2d(layer_input, W_conv) + b_conv)
    h_pool = max_pool_2x2(h_conv)
    return h_pool


def dense_layer(layer_input):
    W_fc = weight_variable([7 * 7 * 64, 1024])
    b_fc = bias_variable([1024])

    h_pool2_flat = tf.reshape(layer_input, [-1, 7*7*64])
    h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
    return h_fc


class ConvNN(object):
    def __init__(self, learning_rate=0.01):
        """ init the model with hyper-parameters etc """
        self.learning_rate = learning_rate

    def inference(self):
        """ This is the forward calculation from x to y """
        self.input = tf.placeholder(tf.float32, [None, 784])
        self.labels = tf.placeholder(tf.float32, [None, 10])

        reshaped_input = tf.reshape(self.input, [-1, 28, 28, 1])

        first_layer_output = conv_layer(reshaped_input,
                                        [5, 5, 1, 32], [32])

        second_layer_output = conv_layer(first_layer_output,
                                         [5, 5, 32, 64], [64])

        dense_layer_output = dense_layer(second_layer_output)

        self.keep_prob = tf.placeholder(tf.float32)
        dropped = tf.nn.dropout(dense_layer_output, self.keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        self.predictions = tf.matmul(dropped, W_fc2) + b_fc2

    def loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels, logits=self.predictions)
        self.loss = tf.reduce_mean(cross_entropy)

    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss)

    def build_graph(self):
        self.inference()
        self.loss()
        return self.optimize()

    def validate(self):
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1),
                                      tf.argmax(self.labels, 1))

        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
