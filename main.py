import os
from model import SoftMax
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Initialise model
    model = SoftMax(learning_rate=0.01)
    train_step = model.build_graph()

    # Initialise Tensorflow session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Run training epochs
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        feed_dict = {
            model.input: batch_xs,
            model.labels: batch_ys
        }
        sess.run(train_step, feed_dict=feed_dict)

    # Validate the model
    feed_dict = {
        model.input: mnist.test.images,
        model.labels: mnist.test.labels
    }

    result = sess.run(model.validate(), feed_dict=feed_dict)

    print('Accuracy: {0:.5f}'.format(result))


if __name__ == '__main__':
    main()
