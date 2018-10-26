import math
import tensorflow as tf
import numpy as np
import sys

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
