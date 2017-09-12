import tensorflow as tf


class SoftMax(object):
    def __init__(self, learning_rate=0.01):
        """ init the model with hyper-parameters etc """
        self.learning_rate = learning_rate
        self.global_step = 0

    def inference(self):
        """ This is the forward calculation from x to y """
        self.input = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        self.predictions = tf.nn.softmax(tf.matmul(self.input, W) + b)

    def loss(self):
        self.labels = tf.placeholder(tf.float32, [None, 10])
        self.loss = tf.reduce_mean(
                -tf.reduce_sum(self.labels * tf.log(self.predictions),
                               reduction_indices=[1]))

    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss)
