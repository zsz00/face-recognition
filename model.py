import re
import os
import tensorflow as tf


def get_loss_layer(name):
    def loss_layer(embedding, labels):
        return arcface_loss_layer(
            embedding, labels, params['m'],
            params['s'], params['num_classes']
        )
    return loss_layer


def model_fn(features, labels, mode, params, config):
    """
    This is a function for creating a tensorflow computational graph.
    The function is in the format required by tf.estimator.
    """

    # the base network, returns embeddings
    def backbone(images, is_training):
        return mobilenet_v1_base(
            images, is_training,
            depth_multiplier=params['depth_multiplier']
        )

    # the last layer (usually it is a layer that does
    # final classification) and computation of the loss
    loss_layer = get_loss_layer(params['loss_to_use'])

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    # features are just a tensor of RGB images
    embedding = backbone(features, is_training)

    # you can use a pretrained backbone network
    if params['pretrained_checkpoint'] is not None:
        with tf.name_scope('init_from_checkpoint'):
            tf.train.init_from_checkpoint(
                params['pretrained_checkpoint'],
                {'MobilenetV1/': 'MobilenetV1/'}
            )

    if not is_training:
        predictions = {
            'embedding': embedding,
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # this is required for exporting a savedmodel
        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name)
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )

    with tf.name_scope('loss_layer'):
        loss = loss_layer(embedding, labels)
        tf.losses.add_loss(loss)
        tf.summary.scalar('loss', loss)

        # add L2 regularization
        with tf.name_scope('weight_decay'):
            add_weight_decay(params['weight_decay'])
            regularization_loss = tf.losses.get_regularization_loss()

        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.EVAL:

        with tf.name_scope('evaluator'):
            evaluator = Evaluator(num_classes=params['num_classes'])
            eval_metric_ops = evaluator.get_metric_ops(filenames, labels, predictions)

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params['lr_boundaries'], params['lr_values'])
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # you can freeze some variables
        trainable_var = tf.trainable_variables()
        regexp = re.compile(params['freeze'])
        var_list = [v for v in trainable_var if not bool(regexp.search(v.name))]

        grads_and_vars = optimizer.compute_gradients(total_loss, var_list=var_list)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    """Add L2 regularization to all (or some) trainable kernel weights."""
    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )
    trainable_vars = tf.trainable_variables()
    kernels = [v for v in trainable_vars if 'weights' in v.name and 'depthwise_weights' not in v.name]
    for K in kernels:
        x = tf.multiply(weight_decay, tf.nn.l2_loss(K))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, x)
