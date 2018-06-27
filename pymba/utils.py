import math
import tensorflow as tf
import numpy as np
import sys

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


class Silence:
    """Suppress any printing while in context"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
