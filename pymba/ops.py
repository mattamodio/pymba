import tensorflow as tf
from .utils import lrelu

def unet_conv(x, nfilt, name, is_training, sn=False, s=2, k=4, batch_norm=None, activation=lrelu):
    x = conv2d(x, nfilt, name=name, s=s, k=k, sn=sn)

    if batch_norm:
        x = batch_norm(x, name='batch_norm_{}'.format(name), is_training=is_training)

    if activation:
        x = activation(x)
    return x

def unet_conv_t(x, encoderx, nfilt, name, is_training, sn=False, skip_connections=True, s=2, k=4, batch_norm=None, activation=tf.nn.relu):
    x = conv2dt(x, nfilt, name=name, s=s, k=k, sn=sn)

    if batch_norm:
        x = batch_norm(x, name='batch_norm_{}'.format(name), is_training=is_training)

    if activation:
        x = activation(x)

    if skip_connections:
        x = tf.concat([x, encoderx], 3)

    return x

def conv2d(x, filters, k=5, s=2, sn=False, name=''):
    weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
    with tf.variable_scope(name):
        if sn:
            w = tf.get_variable("kernel", shape=[k, k, x.get_shape()[-1], filters], initializer=weight_init)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, s, s, 1], padding='SAME')

            bias = tf.get_variable("bias", [filters], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=k, kernel_initializer=weight_init, strides=s, padding='SAME')

        return x

def conv2dt(x, filters, k=5, s=2, padding='SAME', sn=False, name=''):
    weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    with tf.variable_scope(name):
        output_shape = [tf.shape(x)[0], x.get_shape()[1] * s, x.get_shape()[2] * s, filters]
        if sn:
            w = tf.get_variable("kernel", shape=[k, k, filters, x.get_shape()[-1]], initializer=weight_init)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, s, s, 1], padding=padding)

            bias = tf.get_variable("bias", [filters], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=filters, kernel_size=k, kernel_initializer=weight_init, strides=s, padding=padding)

        return x

def dense(x, outdim, sn=False, name=''):
    weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    with tf.variable_scope(name):
        x = tf.layers.flatten(x)
        shape = x.get_shape().as_list()
        indim = shape[-1]
        if sn:
            w = tf.get_variable("kernel", [indim, outdim], tf.float32, initializer=weight_init)
            bias = tf.get_variable("bias", [outdim], initializer=tf.constant_initializer(0.0))
            x = tf.matmul(x, w) + bias
        else:
            x = tf.layers.dense(x, units=outdim, kernel_initializer=weight_init)

        return x

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def batch_norm(x, name, is_training):

    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, scale=True, center=True, is_training=is_training, scope=name, reuse=tf.AUTO_REUSE)

def minibatch(input_, num_kernels=15, kernel_dim=10, name='',):
    """Add minibatch features to input."""
    with tf.variable_scope(name):
        W = tf.get_variable('{}/Wmb'.format(name), [input_.get_shape()[-1], num_kernels * kernel_dim])
        b = tf.get_variable('{}/bmb'.format(name), [num_kernels * kernel_dim])

    x = tf.matmul(input_, W) + b
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_mean(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_mean(tf.exp(-abs_diffs), 2)

    return tf.concat([input_, minibatch_features], axis=-1)

