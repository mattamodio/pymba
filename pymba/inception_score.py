import os.path
import sys
import tarfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
import math
import time
import os
import scipy.linalg


MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_features(sess, images, splits=10, batch_size=32, FID=False, limit_gpu_fraction=.4):
    assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    if softmax is None:
        print("initializing inception model")
        _init_inception(sess)
        print('done initializing inception model')
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    # with tf.Session(config=pymba.build_config(limit_gpu_fraction=limit_gpu_fraction)) as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(batch_size)))
    t = time.time()
    for i in range(n_batches):
        if i % 10 == 0:
            print("{}/{} {:.3f} s".format(i, n_batches, time.time() - t))
            t = time.time()
        inp = inps[(i * batch_size):min((i + 1) * batch_size, len(inps))]
        inp = np.concatenate(inp, 0)
        # pred = sess.run(softmax, {'ExpandDims:0': inp})
        pred = sess.run(softmax, {'InputTensor:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)

    if FID:
        return preds.mean(axis=0), np.cov(preds)

    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception(sess):
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='InputTensor')
    tf.import_graph_def(graph_def, name='', input_map={'ExpandDims:0': input_tensor})
    # Works with an arbitrary minibatch size.
    # with tf.Session(config=pymba.build_config(limit_gpu_fraction=.5)) as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)


def FID(realfeatures, generatedfeatures):
    score = np.sum((realfeatures[0] - generatedfeatures[0])**2) + np.trace(realfeatures[1] + generatedfeatures[1] - 2 * scipy.linalg.sqrtm(realfeatures[1].dot(generatedfeatures[1])))
    score = score.real
    return score
