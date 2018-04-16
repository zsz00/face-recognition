import tensorflow as tf
import math


EPSILON = 1e-8


def arcface_loss_layer(x, labels, m, s, num_classes):
    """
    Arguments:
        x: a float tensor with shape [batch_size, embedding_dimension].
        labels: an int tensor with shape [batch_size].
        m: a float number.
        s: a float number.
        num_classes: an integer.
    Returns:
        a float tensor with shape [].
    """
    embedding_dimension = x.shape.as_list()[1]
    W = tf.get_variable(
        'weights', [embedding_dimension, num_classes], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    W_norm = tf.norm(W, axis=0, keepdims=True) + EPSILON  # shape [1, num_classes]
    x_norm = tf.norm(x, axis=1, keepdims=True) + EPSILON  # shape [batch_size, 1]
    W_normed = tf.divide(W, W_norm)
    x_normed = tf.divide(x, x_norm)
    cos = tf.matmul(x_normed, W_normed)  # shape [batch_size, num_classes]

    batch_size = tf.shape(x)[0]
    indices = tf.stack([tf.range(batch_size, dtype=tf.int32), labels], axis=1)  # shape [batch_size, 2]
    selected_cos = tf.gather_nd(cos, indices)  # shape [batch_size]

    # remember that cos^2(x) + sin^2(x) = 1,
    # so sin(x) = sqrt(1 - cos^2(x)) for 0 < x < PI
    cos2 = tf.pow(selected_cos, 2)
    sin = tf.sqrt(tf.clip_by_value(1.0 - cos2, EPSILON, 1.0))

    cos_m, sin_m = math.cos(m), math.sin(m)
    cos_with_margin = selected_cos * cos_m - sin * sin_m  # cos(theta + m), shape [batch_size]

    mask = tf.one_hot(labels, depth=num_classes, on_value=0, off_value=1, axis=1, dtype=tf.float32)
    # it has shape [batch_size, num_classes]

    logits = s * tf.add(cos * mask, tf.scatter_nd(indices, cos_with_margin, tf.shape(cos)))
    # it has shape [batch_size, num_classes]

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits), axis=0)
    return loss
