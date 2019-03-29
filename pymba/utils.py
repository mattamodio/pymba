import math
import tensorflow as tf
import numpy as np
import sys
import os
import sklearn
import sklearn.metrics
import datetime

def add_to_npzfile(fn, var_to_add, varname_to_add):
    varvals = {}
    if os.path.exists(fn):
        with open(fn, 'rb') as f:
            npzfile = np.load(f)
            varvals = {v: npzfile[v] for v in npzfile.files}

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

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def build_config(limit_gpu_fraction=0.2, limit_cpu_fraction=10):
    if limit_gpu_fraction > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
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


def get_layer(sess, intensor, data, outtensor, batch_size=100):
    out = []
    for batch in np.array_split(data, data.shape[0]/batch_size):
        feed = {intensor: batch}
        batchout = sess.run(outtensor, feed_dict=feed)
        out.append(batchout)
    out = np.concatenate(out, axis=0)

    return out


def mmd(x1, x2):
    def calculate_mmd(k1, k2, k12):
        return k1.sum()/(k1.shape[0]*k1.shape[1]) + k2.sum()/(k2.shape[0]*k2.shape[1]) - 2*k12.sum()/(k12.shape[0]*k12.shape[1])

    k1 = sklearn.metrics.pairwise.pairwise_distances(x1, x1)
    k2 = sklearn.metrics.pairwise.pairwise_distances(x2, x2)
    k12 = sklearn.metrics.pairwise.pairwise_distances(x1, x2)

    mmd = 0
    for sigma in [.01, .1, 1., 10.]:
        k1_ = np.exp(-k1 / sigma**2)
        k2_ = np.exp(-k2 / sigma**2)
        k12_ = np.exp(-k12 / sigma**2)

        mmd += calculate_mmd(k1_, k2_, k12_)

    return mmd

def nameop(op, name):
    return tf.identity(op, name=name)

class Silencer(object):
    def flush(self):
        pass

    def write(self, s):
        pass

class Silence:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = Silencer()
        sys.stderr = Silencer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
