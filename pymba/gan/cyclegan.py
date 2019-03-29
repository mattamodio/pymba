import tensorflow as tf
import os
import pymba
from pymba import lrelu, nameop, tbn, obn
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

class CycleGAN(object):
    def __init__(self,
        args,
        x1,
        x2):
        """Initialize the model."""
        self.args = args
        self.x1 = x1
        self.x2 = x2

        if self.args.restore_folder:
            self._restore(self.args.restore_folder, self.args.limit_gpu_fraction)
            return

        self.iteration = 0

        tffnsb1 = tf.data.Dataset.from_tensor_slices(x1)
        tffnsb2 = tf.data.Dataset.from_tensor_slices(x2)
        tfdatasetb1 = tffnsb1.map(lambda tmp: load_and_preprocess_image(tmp, imdim=args.imdim, channels=args.channels1)).repeat().shuffle(1000).batch(args.batch_size)
        tfdatasetb2 = tffnsb2.map(lambda tmp: load_and_preprocess_image(tmp, imdim=args.imdim, channels=args.channels2)).repeat().shuffle(1000).batch(args.batch_size)
        tfiteratorb1 = tfdatasetb1.make_one_shot_iterator()
        tfiteratorb2 = tfdatasetb2.make_one_shot_iterator()

        nextb1 = tfiteratorb1.get_next()
        nextb2 = tfiteratorb2.get_next()
        nextb1 = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), nextb1)
        nextb2 = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), nextb2)

        self.xb1 = tf.placeholder_with_default(nextb1, shape=[None, args.imdim, args.imdim, args.channels1], name='xb1')
        self.xb2 = tf.placeholder_with_default(nextb2, shape=[None, args.imdim, args.imdim, args.channels2], name='xb2')

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

        savefile = os.path.join(folder, 'CycleGAN')
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

    def _build_loss(self):
        """Collect both of the losses."""
        self._build_loss_D()
        self._build_loss_G()

    def _build_loss_D(self):
        """Discriminator loss."""
        self.loss_Dreal = tf.reduce_mean(adversarial_loss(self.D1_probs_z, label=1))
        self.loss_Dreal += tf.reduce_mean(adversarial_loss(self.D2_probs_z, label=1))

        self.loss_Dfake = tf.reduce_mean(adversarial_loss(self.D1_probs_G, label=0))
        self.loss_Dfake += tf.reduce_mean(adversarial_loss(self.D2_probs_G, label=0))

        self.loss_Dreal = nameop(self.loss_Dreal, 'loss_Dreal')
        tf.add_to_collection('losses', self.loss_Dreal)

        self.loss_Dfake = nameop(self.loss_Dfake, 'loss_Dfake')
        tf.add_to_collection('losses', self.loss_Dfake)

    def _build_optimization(self):
        """Build optimization components."""
        Gvars = [tv for tv in tf.global_variables() if 'G12' in tv.name or 'G21' in tv.name]
        Dvars = [tv for tv in tf.global_variables() if 'D1' in tv.name or 'D2' in tv.name]

        optG = tf.train.AdamOptimizer(self.lr, beta1=.5, beta2=.9)
        self.train_op_G = optG.minimize(self.loss_G, var_list=Gvars, name='train_op_G')

        optD = tf.train.AdamOptimizer(self.lr, beta1=.5, beta2=.9)
        self.train_op_D = optD.minimize(self.loss_Dreal + self.loss_Dfake, var_list=Dvars, name='train_op_D')

    def train(self):
        """Take a training step with batches from each domain."""
        self.iteration += 1

        feed = {tbn('lr:0'): self.args.learning_rate,
                tbn('is_training:0'): True}

        self.sess.run([obn('train_op_D')], feed_dict=feed)
        self.sess.run([obn('train_op_G')], feed_dict=feed)

    def get_layer(self, xb1, xb2, name):
        """Get a layer of the network by name for the entire datasets given in xb1 and xb2."""
        tensor_name = "{}:0".format(name)
        tensor = tbn(tensor_name)

        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('is_training:0'): False}

        layer = self.sess.run(tensor, feed_dict=feed)

        return layer

    def get_loss_names(self):
        """Return a string for the names of the loss values."""
        losses = [tns.name[:-2].replace('loss_', '').split('/')[-1] for tns in tf.get_collection('losses') if 'hinge' not in tns.name]
        return "Losses: {}".format(' '.join(losses))

    def get_loss(self, xb1, xb2):
        """Return all of the loss values for the given input."""
        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('is_training:0'): False}

        losses = self.sess.run(tf.get_collection('losses'), feed_dict=feed)

        lstring = ' '.join(['{:.3f}'.format(loss) for loss in losses])

        return lstring

    def _build(self):
        self.G12 = Generator(args=self.args, name='G12')
        self.Gb2 = self.G12(self.xb1, is_training=self.is_training)
        self.Gb2 = nameop(self.Gb2, 'Gb2')

        self.G21 = Generator(args=self.args, name='G21')
        self.Gb1 = self.G21(self.xb2, is_training=self.is_training)
        self.Gb1 = nameop(self.Gb1, 'Gb1')

        self.Gb2_reconstructed = self.G12(self.Gb1, is_training=self.is_training)
        self.Gb1_reconstructed = self.G21(self.Gb2, is_training=self.is_training)
        self.Gb1_reconstructed = nameop(self.Gb1_reconstructed, 'Gb1_reconstructed')
        self.Gb2_reconstructed = nameop(self.Gb2_reconstructed, 'Gb2_reconstructed')

        if self.args.channels1 != self.args.channels2:
            self.G21_identity = tf.zeros_like(self.xb1)
            self.G12_identity = tf.zeros_like(self.xb2)
        else:
            self.G12_identity = self.G12(self.xb2, is_training=self.is_training)
            self.G21_identity = self.G21(self.xb1, is_training=self.is_training)
            self.G21_identity = nameop(self.G21_identity, 'output_xb1')
            self.G12_identity = nameop(self.G12_identity, 'output_xb2')

        self.D1 = Discriminator(args=self.args, name='D1')
        self.D2 = Discriminator(args=self.args, name='D2')

        self.D1_probs_z = self.D1(self.xb1, is_training=self.is_training)
        self.D1_probs_G = self.D1(self.Gb1, is_training=self.is_training)

        self.D2_probs_z = self.D2(self.xb2, is_training=self.is_training)
        self.D2_probs_G = self.D2(self.Gb2, is_training=self.is_training)

    def _build_loss_G(self):
        # fool the discriminator loss
        self.loss_G1_discr = tf.reduce_mean(adversarial_loss(self.D1_probs_G, label=1))
        self.loss_G2_discr = tf.reduce_mean(adversarial_loss(self.D2_probs_G, label=1))
        tf.add_to_collection('losses', nameop(self.loss_G1_discr, 'loss_G1_discr'))
        tf.add_to_collection('losses', nameop(self.loss_G2_discr, 'loss_G2_discr'))
        self.loss_G = self.args.lambda_adversary * (self.loss_G1_discr + self.loss_G2_discr)

        # reconstruction loss
        if self.args.lambda_cycle:
            self.loss_G1_recon = tf.reduce_mean((self.xb1 - self.Gb1_reconstructed)**2)
            self.loss_G2_recon = tf.reduce_mean((self.xb2 - self.Gb2_reconstructed)**2)
            tf.add_to_collection('losses', nameop(self.loss_G1_recon, 'loss_G1_recon'))
            tf.add_to_collection('losses', nameop(self.loss_G2_recon, 'loss_G2_recon'))
            self.loss_G += self.args.lambda_cycle * (self.loss_G1_recon + self.loss_G2_recon)

        # identity mapping loss
        if self.args.lambda_identity:
            self.loss_G1_ident = tf.reduce_mean((self.G12_identity - self.xb2)**2)
            self.loss_G2_ident = tf.reduce_mean((self.G21_identity - self.xb1)**2)
            tf.add_to_collection('losses', nameop(self.loss_G1_ident, 'loss_G1_ident'))
            tf.add_to_collection('losses', nameop(self.loss_G2_ident, 'loss_G2_ident'))
            self.loss_G += self.args.lambda_identity * (self.loss_G1_ident + self.loss_G2_ident)


class Generator(object):
    def __init__(self,
        args,
        name=''):
        self.args = args
        self.name = name
        self.first_call = True

    def __call__(self, x, is_training):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.first_call: print(tf.get_variable_scope().name)

            # up
            encoders = []
            nshape = x.get_shape()[1].value
            filt = self.args.nfilt
            layer = 1
            iinput = x
            minshape = 8
            basefilt = filt
            maxfilt = 4 * filt
            nfilt = basefilt

            while nshape > minshape:
                # pre
                if self.first_call: print(iinput)
                # layer
                output = unet_conv(iinput, nfilt, 'e{}'.format(layer), is_training, activation=lrelu, batch_norm=batch_norm if layer != 1 else False)
                # post
                encoders.append([output, nshape, nfilt, layer])
                nshape /= 2
                nfilt = min(2 * nfilt, maxfilt)
                layer += 1
                iinput = output

            if self.first_call: print(output)

            iinput = encoders.pop()[0]

            # down
            for encoderinput, nshape, nfilt, layer in encoders[::-1]:
                # layer
                output = unet_conv_t(iinput, encoderinput, nfilt, 'd{}'.format(layer), is_training, activation=tf.nn.relu, batch_norm=batch_norm)
                # post
                iinput = output
                if self.first_call: print(iinput)

            if '12' in self.name:
                outd = self.args.channels2
            else:
                outd = self.args.channels1
            # out
            out = unet_conv_t(output, None, outd, 'out', is_training, activation=tf.nn.tanh, batch_norm=False, skip_connections=False)

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

















