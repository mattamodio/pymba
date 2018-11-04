import numpy as np
import matplotlib.pyplot as plt
import glob


def get_data_cifar(datadir, label1, label2, tanh_transform=True):
    #     classes
    #         0: car
    #         1: plane
    #         2: bird
    #         3: cat
    #         4: deer
    #         5: dog
    #         6: frog
    #         7: horse
    #         8: ship
    #         9: truck

    with open('{}/cifar.npz'.format(datadir), 'rb') as f:
        npzfile = np.load(f)
        data = npzfile['data'].astype(np.float32)
        labels = npzfile['labels']

    if tanh_transform:
        data = (data / 127.5) - 1

    batch1 = data[labels == label1]
    batch2 = data[labels == label2]

    return batch1, batch2


def get_data_imagenet(datadir, wnid1, wnid2, D=128, tanh_transform=True):
    b1 = sorted(glob.glob('{}/{}/*.JPEG'.format(datadir, wnid1)))
    b2 = sorted(glob.glob('{}/{}/*.JPEG'.format(datadir, wnid2)))

    b1 = [scipy.misc.imresize(plt.imread(f), (D, D)) for f in b1]
    b1 = [img for img in b1 if len(img.shape) == 3]
    b1 = np.stack(b1, axis=0)

    b2 = [scipy.misc.imresize(plt.imread(f), (D, D)) for f in b2]
    b2 = [img for img in b2 if len(img.shape) == 3]
    b2 = np.stack(b2, axis=0)

    b1 = b1.astype(np.float32)
    b2 = b2.astype(np.float32)

    if tanh_transform:
        b1 = (b1 / 127.5) - 1
        b2 = (b2 / 127.5) - 1

    return b1, b2
