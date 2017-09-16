import tensorflow as tf
from datetime import datetime


def train_model(sess, model, train_data, n_batches=10000, batch_size=100):

    train_step = model.build_graph()

    file_writer, loss_summary = init_logging(model)

    tf.global_variables_initializer().run()

    # Run training epochs
    for batch_index in range(n_batches):
        batch_xs, batch_ys = train_data.next_batch(batch_size)

        feed_dict = {
            model.input: batch_xs,
            model.labels: batch_ys,
            model.keep_prob: 0.5
        }

        if batch_index % 100 == 0:
            summary_str = loss_summary.eval(feed_dict=feed_dict)
            step = batch_index * batch_size
            file_writer.add_summary(summary_str, step)

        sess.run(train_step, feed_dict=feed_dict)


def evaluate_model(sess, model, test_data):
    # Validate the model
    feed_dict = {
        model.input: test_data.images,
        model.labels: test_data.labels,
        model.keep_prob: 1
    }

    result = sess.run(model.validate(), feed_dict=feed_dict)
    print('Accuracy: {0:.5f}'.format(result))


def init_logging(model):
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = 'tf_logs'
    logdir = '{}/run-{}/'.format(root_logdir, now)

    loss_summary = tf.summary.scalar('Loss', model.loss)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    return file_writer, loss_summary
