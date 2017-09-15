import tensorflow as tf
from datetime import datetime


def train_model(sess, model, train_data):

    train_step = model.build_graph()

    init_logging()

    tf.global_variables_initializer().run()

    # Run training epochs
    for _ in range(1000):
        batch_xs, batch_ys = train_data.next_batch(100)
        feed_dict = {
            model.input: batch_xs,
            model.labels: batch_ys
        }
        sess.run(train_step, feed_dict=feed_dict)


def evaluate_model(sess, model, test_data):
    # Validate the model
    feed_dict = {
        model.input: test_data.images,
        model.labels: test_data.labels
    }

    result = sess.run(model.validate(), feed_dict=feed_dict)
    print('Accuracy: {0:.5f}'.format(result))


def init_logging():
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = 'tf_logs'
    logdir = '{}/run-{}/'.format(root_logdir, now)

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
