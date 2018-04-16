import tensorflow as tf


EPSILON = 1e-8


def a_softmax_loss_layer(x, y, alpha, num_classes):
    """
    Arguments:
        x: a float tensor with shape [batch_size, embedding_dimension].
        y: an int tensor with shape [batch_size].
        alpha: a float number.
        num_classes: an integer.
    Returns:
        a float tensor with shape [].
    """
    batch_size = tf.shape(x)[0]
    embedding_dimension = x.shape.as_list()[1]
    W = tf.get_variable(
        'W', [embedding_dimension, num_classes], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    W_norm = tf.norm(W, axis=0) + EPSILON  # shape [num_classes]
    logits = tf.divide(tf.matmul(x, W), W_norm)  # shape [batch_size, num_classes]

    indices = tf.stack([tf.range(batch_size, dtype=tf.int32), y], axis=1)  # shape [batch_size, 2]
    selected_logits = tf.gather_nd(logits, indices)  # shape [batch_size]

    x_norm = tf.norm(x, axis=1) + EPSILON  # shape [batch_size]
    cos = tf.divide(selected_logits, x_norm)  # shape [batch_size]

    # remember that cos(2x) = 2cos^2(x) - 1,
    # so cos(4x) = 8(cos^4(x)-cos^2(x)) + 1
    cos2 = tf.pow(cos, 2)
    cos4 = tf.pow(cos, 4)

    # assume that cos = cos(angle) and 0 < angle < PI
    sign0 = tf.sign(cos)  # 1 if (angle < PI/2)
    sign1 = tf.sign(2*cos2 - 1)  # 1 if (angle < PI/4) or (3*PI/4 < angle)
    sign2 = tf.multiply(sign1, sign0)  # 1 if (angle < PI/4) or (PI/2 < angle < 3*PI/4)

    sign3 = 2*sign0 + sign2 - 3  # 2*(sign0 - 1) + (sign2 - 1)
    # 0 if (angle < PI/4)
    # -2 if (PI/4 < angle < PI/2)
    # -4 if (PI/2 < angle < 3*PI/4)
    # -6 if (3*PI/4 < angle)

    psi_theta = sign2*(8*cos4 - 8*cos2 + 1) + sign3  # shape [batch_size]
    new_logits = tf.multiply(psi_theta, x_norm)  # shape [batch_size]

    mask = tf.one_hot(y, depth=num_classes, on_value=0, off_value=1, axis=1, dtype=tf.float32)
    # it has shape [batch_size, num_classes]

    combined_logits = tf.add(logits * mask, tf.scatter_nd(indices, new_logits, tf.shape(logits)))
    updated_logits = (alpha/(1.0 + alpha)) * logits + (1.0/(1.0 + alpha)) * combined_logits
    # they have shape [batch_size, num_classes]

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=updated_logits), axis=0)
    return loss
