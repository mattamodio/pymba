import tensorflow as tf

def adversarial_loss(logits, label, gantype='gan'):
    if gantype == 'gan':
        if label == 0:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
        elif label == 1:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
        else:
            raise Exception("Label not in {0,1} for adversarial loss")
    elif gantype == 'wgan':
        if label == 0:
            return -logits
        elif label == 1:
            return logits
        else:
            raise Exception("Label not in {0,1} for adversarial loss")
    elif gantype == 'hinge':
        if label == 0:
            return tf.nn.relu(1. + logits)
        elif label == 1:
            return tf.nn.relu(1. - logits)
        else:
            raise Exception("Label not in {0,1} for adversarial loss")
    else:
        raise Exception("Gantype not implemented")



def discriminatory_gradient_penalty(x, G, Discriminator_fn, **kwargs):
    """
    code adapted from original creator: https://github.com/LynnHo/DCGAN-LSGAN-WGAN-WGAN-GP-Tensorflow/blob/ed45429a295aa6f7ff79b0a26c35a2c0ab0b059a/train_cartoon_wgan_gp.py
    """
    from tensorflow.python.ops import gradients_impl
    from tensorflow.python.ops.losses import losses
    alpha = tf.random_uniform(shape=[tf.shape(x)[0], 1, 1, 1], minval=0., maxval=1., name='alpha')
    interp = x + alpha * (G - x)
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        vec_interp = Discriminator_fn(interp, **kwargs)


    gradients = gradients_impl.gradients(vec_interp, interp)[0]
    gradient_squares = tf.reduce_sum(gradients**2, axis=list(range(1, gradients.shape.ndims)))

    slopes = tf.sqrt(gradient_squares + 1e-6)
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

    return gradient_penalty