import tensorflow as tf
from models.layers import conv_layer, output_layer, dense_layer


class ConvNN(object):
    def __init__(self, learning_rate=0.05):
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

        self.predictions = output_layer(dropped)

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
