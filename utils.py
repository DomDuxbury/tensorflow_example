import tensorflow as tf


def train_model(sess, model, train_data):
    train_step = model.build_graph()
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
