import math
import tensorflow as tf
import numpy as np
import sys
import os

def add_to_npzfile(fn, var_to_add, varname_to_add):
    varvals = {}
    if os.path.exists(fn):
        with open(fn, 'rb') as f:
            npzfile = np.load(f)
            varvals = {v:npzfile[v] for v in npzfile.files}

    varvals[varname_to_add] = var_to_add

    with open(fn, 'wb+') as f:
        np.savez(f, **varvals)

def asinh(x, scale=5.):
    """Asinh transform."""
    f = np.vectorize(lambda y: math.asinh(y / scale))
    return f(x)

def sinh(x, scale=5.):
    """Reverse transform for asinh."""
    return scale * np.sinh(x)

def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky ReLU activation."""
    return tf.maximum(x, leak * x)

def tbn(name):
    """Get the tensor in the default graph of the given name."""
    return tf.get_default_graph().get_tensor_by_name(name)

def obn(name):
    """Get the operation node in the default graph of the given name."""
    return tf.get_default_graph().get_operation_by_name(name)

def now():
    return datetime.datetime.now().strftime('%m-%d %H:%M:%S')

def get_all_node_names():
    return [n.name for n in tf.get_default_graph().as_graph_def().node]


def conv(x, nfilt, name, padding='same', k=4, s=2, d=1, use_bias=True):
    return tf.layers.conv2d(x, filters=nfilt, kernel_size=k, padding=padding, strides=[s,s], dilation_rate=[d,d],
                            kernel_initializer=tf.truncated_normal_initializer(0,.02), activation=None,
                            use_bias=use_bias, name=name)

def conv_t(x, nfilt, name, padding='same', k=4, s=2, use_bias=True):
    return tf.layers.conv2d_transpose(x, filters=nfilt, kernel_size=k, padding=padding, strides=[s,s], 
                            kernel_initializer=tf.truncated_normal_initializer(0,.02), activation=None,
                            use_bias=use_bias, name=name)

def unet_conv(x, nfilt, name, is_training, s=2, k=4, d=1, use_bias=True, batch_norm=None, activation=lrelu):
    x = conv(x, nfilt, name, use_bias=use_bias, d=d, s=s, k=k)
    if batch_norm:
        x = batch_norm(x, name='batch_norm_{}'.format(name), training=is_training)

    if activation:
        x = activation(x)
    return x

def unet_conv_t(x, encoderx, nfilt, name, is_training, skip_connections=True, s=2, k=4, use_bias=True, use_dropout=0, batch_norm=None, activation=tf.nn.relu):
    x = conv_t(x, nfilt, name, s=s, k=k, use_bias=use_bias)
    if use_dropout:
        x = tf.layers.dropout(x, use_dropout, training=is_training)

    if batch_norm:
        x = batch_norm(x, name='batch_norm_{}'.format(name), training=is_training)

    if activation:
        x = activation(x)

    if skip_connections:
        x = tf.concat([x,encoderx], 3)
    
    return x

def batch_normalization(tensor, name, training):

    return tf.layers.batch_normalization(tensor, training=training, momentum=.9, scale=True, fused=True, name=name)

def build_config(limit_gpu_fraction=0.2, limit_cpu_fraction=10):
    if limit_gpu_fraction > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu_options = tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=limit_gpu_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config = tf.ConfigProto(device_count={'GPU': 0})
    if limit_cpu_fraction is not None:
        if limit_cpu_fraction <= 0:
            # -2 gives all CPUs except 2
            cpu_count = min(
                1, int(os.cpu_count() + limit_cpu_fraction))
        elif limit_cpu_fraction < 1:
            # 0.5 gives 50% of available CPUs
            cpu_count = min(
                1, int(os.cpu_count() * limit_cpu_fraction))
        else:
            # 2 gives 2 CPUs
            cpu_count = int(limit_cpu_fraction)
        config.inter_op_parallelism_threads = cpu_count
        config.intra_op_parallelism_threads = cpu_count
        os.environ['OMP_NUM_THREADS'] = str(1)
        os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    return config


class Silencer(object):
    def flush(self): pass
    def write(self, s): pass

class Silence:
    """Suppress any printing while in context"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = Silencer()
        sys.stderr = Silencer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
