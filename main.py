import os
from model import SoftMax
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    model = SoftMax(learning_rate=0.01)
    model.inference()
    model.loss()
    train_step = model.optimize()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={model.input: batch_xs,
                                        model.labels: batch_ys})

    correct_prediction = tf.equal(tf.argmax(model.predictions, 1),
                                  tf.argmax(model.labels, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={model.input: mnist.test.images,
                                        model.labels: mnist.test.labels}))


if __name__ == '__main__':
    main()
