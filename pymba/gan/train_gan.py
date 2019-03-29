import numpy as np
import glob
import os
import pickle
import itertools
import matplotlib.pyplot as plt
import argparse
import skimage.transform
from pymba import now
from pymba.loader import Loader
from gan import GAN

def parse_args():
    parser = argparse.ArgumentParser()
    # required params
    parser.add_argument('--savefolder', type=str)
    parser.add_argument('--data_dir', type=str)

    # data params
    parser.add_argument('--imdim', type=int, default=64)
    parser.add_argument('--upsamplerate', type=float, default=1.2)
    parser.add_argument('--channels', type=int, default=3)

    # model params
    parser.add_argument('--nfilt', type=int, default=64)
    parser.add_argument('--dimz', type=int, default=100)

    # training params
    parser.add_argument('--training_steps', type=int, default=300000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--limit_gpu_fraction', type=float, default=.4)
    parser.add_argument('--restore_folder', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=.0002)

    args = parser.parse_args()
    if args.savefolder and not os.path.exists(args.savefolder): os.mkdir(args.savefolder)

    return args

args = parse_args()

if not args.restore_folder:
    with open(os.path.join(args.savefolder, 'args.txt'), 'w+') as f:
        for arg in vars(args):
            argstring = "{}: {}\n".format(arg, vars(args)[arg])
            f.write(argstring)
            print(argstring[:-1])
    with open(os.path.join(args.savefolder, 'args.pkl'), 'wb+') as f:
        pickle.dump(args, f)


if not os.path.exists("{}/output".format(args.savefolder)): os.mkdir("{}/output".format(args.savefolder))

def glob_fns(directory):
    fn_extensions = [[s.lower(), s.upper()] for s in ['jpeg', 'jpg', 'png']]
    fn_extensions = list(itertools.chain(*fn_extensions))
    fns = [glob.glob(os.path.join(directory, '*.{}'.format(ext))) for ext in fn_extensions]
    fns = list(itertools.chain(*fns))
    return fns

fns = glob_fns(args.data_dir)

load = Loader(np.array(fns), shuffle=True)

print("Number of images: {}".format(len(fns)))

model = GAN(args, x=fns)

plt.ioff(); fig = plt.figure(figsize=(6, 6))
np.set_printoptions(precision=3)

for i in range(1, args.training_steps):
    if i % 10 == 0: print("Iter {} ({})".format(i, now()))
    model.train()

    if i and (i == 50 or i % 500 == 0):
        model.save(folder=args.savefolder)

        z = np.random.uniform(-1, 1, [36, args.dimz])
        test = np.stack([skimage.transform.resize(plt.imread(im), [args.imdim, args.imdim]) for im in load.next_batch(36)], axis=0)

        test = (test / test.max() * 2) - 1
        lstring = model.get_loss(x=test, z=z)

        sample = model.sample(z=z)

        # back to [0,1] for imshow
        test = (test + 1) / 2
        sample = (sample + 1) / 2
        if args.channels == 1:
            test = test[:, :, :, 0]
            sample = sample[:, :, :, 0]

        fig.clf()
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        axes = fig.subplots(6, 6)
        for ii, ax in enumerate(axes.flatten()):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(sample[ii])
        fig.canvas.draw()
        fig.savefig('{}/output/sample.png'.format(args.savefolder), dpi=500)

        print("{} ({}): {}".format(i, now(), lstring))



model.save(folder=args.savefolder)
print("Done! Final model saved.")






















