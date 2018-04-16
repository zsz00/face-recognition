import tensorflow as tf
import math


EPSILON = 1e-8


def center_loss(x, y, alpha, num_classes):
    """
    Arguments:
        x: a float tensor with shape [batch_size, embedding_dimension].
        y: an int tensor with shape [batch_size].
        alpha: a float number.
        num_classes: an integer.
    Returns:
        loss: a float tensor with shape [].
        centers: a float variable with shape [num_classes, embedding_dimension].
        centers_update_op: an op.
    """
    embedding_dimension = x.shape.as_list()[1]
    centers = tf.get_variable(
        'centers', [num_classes, embedding_dimension], dtype=tf.float32,
        initializer=tf.constant_initializer(0.0), trainable=False
    )

    centers_batch = tf.gather(centers, y)  # shape [batch_size, embedding_dimension]
    difference = centers_batch - x
    loss = tf.nn.l2_loss(difference)  # shape []

    _, indices_of_counts, label_counts = tf.unique_with_counts(y)
    # they have shapes [batch_size] and [num_unique]
    appear_times = tf.gather(label_counts, indices_of_counts)  # shape [batch_size]
    appear_times = tf.expand_dims(appear_times, axis=1)

    delta = alpha * tf.divide(difference, tf.to_float(1 + appear_times))  # shape [batch_size, embedding_dimension]
    centers_update_op = tf.scatter_sub(centers, labels, delta)

    return loss, centers, centers_update_op
