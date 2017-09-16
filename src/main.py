import os
import tensorflow as tf
import utils
from tensorflow.examples.tutorials.mnist import input_data
from models.ConvNN import ConvNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    # Load the data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Initialise model
    model = ConvNN(learning_rate=0.01)

    # Initialise Tensorflow session
    sess = tf.InteractiveSession()

    # Train the model
    utils.train_model(sess, model, mnist.train, n_batches=100)

    # Validate the model
    utils.evaluate_model(sess, model, mnist.test)


if __name__ == '__main__':
    main()
