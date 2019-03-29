import tensorflow as tf
import pymba
import os
import glob
from pymba import lrelu, nameop, tbn
from pymba.ops import batch_norm, unet_conv, unet_conv_t, dense, spectral_norm
from pymba.gan.ops import adversarial_loss


def preprocess_image(image, imdim, channels=3):
    image = tf.image.decode_jpeg(image, channels=channels)
    image = tf.image.resize_images(image, [imdim, imdim])
    image = (image / tf.reduce_max(image)) * 2 - 1

    return image

def load_and_preprocess_image(path, imdim, channels):
    image = tf.read_file(path)
    return preprocess_image(image, imdim, channels)


class GAN(object):
    def __init__(self, args, x):
        self.args = args

        if self.args.restore_folder:
            self._restore(self.args.restore_folder, self.args.limit_gpu_fraction)
            return

        self.iteration = 0

        tffns = tf.data.Dataset.from_tensor_slices(x)
        tfdataset = tffns.map(lambda tmp: load_and_preprocess_image(tmp, imdim=args.imdim, channels=args.channels)).repeat().shuffle(1000).batch(args.batch_size)
        tfiterator = tfdataset.make_one_shot_iterator()

        next = tfiterator.get_next()
        next = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), next)

        self.x = tf.placeholder_with_default(next, shape=[None, args.imdim, args.imdim, args.channels], name='x')
        z = tf.random.uniform(shape=[tf.shape(self.x)[0], args.dimz], minval=-1., maxval=1.)
        self.z = tf.placeholder_with_default(z, shape=[None, args.dimz], name='z')

        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        self._build()
        self._build_loss()
        self._build_optimization()

        self.init_session(limit_gpu_fraction=self.args.limit_gpu_fraction)
        self.graph_init(self.sess)

    def init_session(self, limit_gpu_fraction=.4, no_gpu=False):
        """Initialize the session."""
        if no_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
        elif limit_gpu_fraction:
            self.sess = tf.Session(config=pymba.build_config(limit_gpu_fraction=limit_gpu_fraction))
        else:
            self.sess = tf.Session()

    def graph_init(self, sess=None):
        """Initialize graph variables."""
        if not sess: sess = self.sess

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())

        return self.saver

    def save(self, iteration=None, saver=None, sess=None, folder=None):
        """Save the model."""
        if not iteration: iteration = self.iteration
        if not saver: saver = self.saver
        if not sess: sess = self.sess
        if not folder: folder = self.save_folder

        savefile = os.path.join(folder, 'GAN')
        saver.save(sess, savefile, write_meta_graph=True)
        print("Model saved to {}".format(savefile))

    def _restore(self, restore_folder, limit_gpu_fraction, no_gpu=False):
        """Restore the model from a saved checkpoint."""
        tf.reset_default_graph()
        self.init_session(limit_gpu_fraction, no_gpu)
        ckpt = tf.train.get_checkpoint_state(restore_folder)
        self.saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        self.iteration = 0
        print("Model restored from {}".format(restore_folder))

    def _build(self):
        G = Generator(args=self.args, name='generator')
        D = Discriminator(args=self.args, name='discriminator')

        self.samples = G(self.z, is_training=self.is_training)
        nameop(self.samples, 'samples')


        self.D_fake = D(self.samples, is_training=self.is_training)
        self.D_real = D(self.x, is_training=self.is_training)

    def _build_loss(self, gantype='gan'):
        self.loss_Dfake = tf.reduce_mean(adversarial_loss(logits=self.D_fake, label=0, gantype=gantype))
        self.loss_Dreal = tf.reduce_mean(adversarial_loss(logits=self.D_real, label=1, gantype=gantype))
        self.loss_D = self.loss_Dfake + self.loss_Dreal

        self.loss_G = tf.reduce_mean(adversarial_loss(logits=self.D_fake, label=1, gantype=gantype))

    def _build_optimization(self):
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        #####
        # optimization for D
        dvars = [tv for tv in tf.trainable_variables() if 'discriminator' in tv.name]
        optD = tf.train.AdamOptimizer(self.learning_rate, beta1=.5)
        self.update_op_D = optD.minimize(self.loss_D, var_list=dvars)

        #####
        # optimization for G
        gvars = [tv for tv in tf.trainable_variables() if 'generator' in tv.name]
        optG = tf.train.AdamOptimizer(self.learning_rate, beta1=.5)
        self.update_op_G = optG.minimize(self.loss_G, var_list=gvars)

    def train(self, x=None, z=None, learning_rate=.001):
        feed = {}
        if x:
            feed[tbn('x:0')] = x
        if z:
            feed[tbn('z:0')] = z
        feed[tbn('learning_rate:0')] = learning_rate
        feed[tbn('is_training:0')] = True

        self.sess.run(self.update_op_D, feed_dict=feed)
        self.sess.run(self.update_op_G, feed_dict=feed)

    def sample(self, z):

        feed = {}
        feed[tbn('is_training:0')] = False
        feed[tbn('z:0')] = z
        out = self.sess.run(tbn('samples:0'), feed_dict=feed)

        return out

    def get_loss(self, x, z):
        feed = {}
        feed[tbn('x:0')] = x
        feed[tbn('z:0')] = z
        feed[tbn('is_training:0')] = False

        return self.sess.run([self.loss_G, self.loss_Dreal, self.loss_Dfake], feed_dict=feed)










class Generator(object):
    def __init__(self,
        args,
        name=''):
        self.args = args
        self.name = name
        self.first_call = True

    def __call__(self, z, is_training):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.first_call: print(tf.get_variable_scope().name)

            shape = 4
            nfilt = self.args.nfilt * int((self.args.imdim / shape) // 2)

            projection = dense(z, 4 * 4 * nfilt, name='projection')
            projection = batch_norm(projection, name='projection', is_training=is_training)
            projection = tf.nn.relu(projection)
            iinput = tf.reshape(projection, [-1, shape, shape, nfilt])
            nfilt = nfilt // 2

            layer = 1
            while shape < self.args.imdim // 2:
                if self.first_call: print(iinput)
                output = unet_conv_t(iinput, None, nfilt, 'generator{}'.format(layer), is_training, activation=tf.nn.relu, batch_norm=batch_norm, skip_connections=False)

                shape *= 2
                nfilt = nfilt // 2
                layer += 1
                iinput = output

            if self.first_call: print(output)

            # out
            out = unet_conv_t(output, None, self.args.channels, 'out', is_training, activation=tf.nn.tanh, batch_norm=False, skip_connections=False)

            if self.first_call:
                print("{}\n".format(out))
                self.first_call = False

        return out

class Discriminator(object):
    """A discriminator class."""

    def __init__(self,
        args,
        name=''):
        """Initialize a new discriminator."""
        self.args = args
        self.name = name
        self.first_call = True

    def __call__(self, x, is_training):
        """Return the output of the discriminator."""
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.first_call: print(tf.get_variable_scope().name)

            minshape = 4
            filt = self.args.nfilt * 2
            maxfilt = 8 * filt
            basefilt = filt
            nfilt = basefilt
            nshape = x.get_shape()[1].value
            layer = 1
            iinput = x

            while nshape > minshape:
                if self.first_call: print(iinput)
                output = unet_conv(iinput, nfilt, 'h{}'.format(layer), is_training, sn=spectral_norm, activation=lrelu, batch_norm=batch_norm if layer != 1 else False)
                nshape /= 2
                nfilt = min(2 * nfilt, maxfilt)
                layer += 1
                iinput = output
            if self.first_call: print(output)


            output = tf.reduce_sum(output, axis=[1, 2])
            out = dense(output, 1, sn=spectral_norm, name='out')

            if self.first_call:
                print("{}\n".format(out))
                self.first_call = False

        return out






















