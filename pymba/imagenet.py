import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import skimage.transform



def get_file(_file):
    _here = os.path.dirname(os.path.abspath(__file__))
    _file = os.path.join(_here, _file)
    return _file


def get_imagenet_classes():
    fn = get_file('imagenet_classes.py')
    classes = {}
    with open(fn) as f:
        for line in f:
            wnid, num, name = line.strip().split(' ')
            classes[name] = wnid
    return classes

def get_imagenet_data(datadir, wnid, dim=128, tanh_transform=True):
    b1 = sorted(glob.glob('{}/{}/*.JPEG'.format(datadir, wnid)))

    b1 = [skimage.transform.resize(plt.imread(f), (dim, dim)) for f in b1]
    b1 = [img for img in b1 if len(img.shape) == 3]
    b1 = np.stack(b1, axis=0)

    b1 = b1.astype(np.float32)

    if tanh_transform:
        b1 = (b1 / 127.5) - 1

    return b1












