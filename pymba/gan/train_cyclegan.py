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
from pymba.gan.cyclegan import CycleGAN

def parse_args():
    parser = argparse.ArgumentParser()
    # required params
    parser.add_argument('--savefolder', type=str)
    parser.add_argument('--data_dirb1', type=str)
    parser.add_argument('--data_dirb2', type=str)

    # data params
    parser.add_argument('--imdim', type=int, default=64)
    parser.add_argument('--upsamplerate', type=float, default=1.2)
    parser.add_argument('--channels1', type=int, default=3)
    parser.add_argument('--channels2', type=int, default=3)

    # model params
    parser.add_argument('--nfilt', type=int, default=64)
    parser.add_argument('--lambda_cycle', type=float, default=1.)
    parser.add_argument('--lambda_identity', type=float, default=.1)
    parser.add_argument('--lambda_adversary', type=float, default=1)

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

fnsb1 = glob_fns(args.data_dirb1)
fnsb2 = glob_fns(args.data_dirb2)

load1 = Loader(np.array(fnsb1), shuffle=True)
load2 = Loader(np.array(fnsb2), shuffle=True)

print("Domain 1 imgs: {}".format(len(fnsb1)))
print("Domain 2 imgs: {}".format(len(fnsb2)))

model = CycleGAN(args, x1=fnsb1, x2=fnsb2)


plt.ioff(); fig = plt.figure(figsize=(4, 10))
np.set_printoptions(precision=3)

for i in range(1, args.training_steps):

    if i % 10 == 0: print("Iter {} ({})".format(i, now()))
    model.train()

    if i == 50 or i % 500 == 0:
        model.save(folder=args.savefolder)

        testb1 = np.stack([skimage.transform.resize(plt.imread(im), [args.imdim, args.imdim]) for im in load1.next_batch(10)], axis=0)
        testb2 = np.stack([skimage.transform.resize(plt.imread(im), [args.imdim, args.imdim]) for im in load2.next_batch(10)], axis=0)

        testb1 = (testb1 / testb1.max() * 2) - 1
        testb2 = (testb2 / testb2.max() * 2) - 1
        lstring = model.get_loss(testb1, testb2)

        Gb1 = model.get_layer(testb1, testb2, name='Gb1')
        Gb2 = model.get_layer(testb1, testb2, name='Gb2')

        # back to [0,1] for imshow
        testb1 = (testb1 + 1) / 2
        testb2 = (testb2 + 1) / 2
        Gb1 = (Gb1 + 1) / 2
        Gb2 = (Gb2 + 1) / 2
        if args.channels1 == 1:
            testb1 = testb1[:, :, :, 0]
            Gb1 = Gb1[:, :, :, 0]
        if args.channels2 == 1:
            testb2 = testb2[:, :, :, 0]
            Gb2 = Gb2[:, :, :, 0]

        fig.clf()
        fig.subplots_adjust(.01, .01, .99, .99, .01, .01)
        for ii in range(10):
            ax1 = fig.add_subplot(10, 2, 2 * ii + 1)
            ax2 = fig.add_subplot(10, 2, 2 * ii + 2)
            for ax in (ax1, ax2):
                ax.set_xticks([])
                ax.set_yticks([])

            ax1.imshow(testb1[ii])
            ax2.imshow(Gb2[ii])
        fig.canvas.draw()
        fig.savefig('{}/output/b1_to_b2.png'.format(args.savefolder), dpi=500)


        fig.clf()
        fig.subplots_adjust(.01, .01, .99, .99, .01, .01)
        for ii in range(10):
            ax1 = fig.add_subplot(10, 2, 2 * ii + 1)
            ax2 = fig.add_subplot(10, 2, 2 * ii + 2)
            for ax in (ax1, ax2):
                ax.set_xticks([])
                ax.set_yticks([])

            ax1.imshow(testb2[ii])
            ax2.imshow(Gb1[ii])
        fig.canvas.draw()
        fig.savefig('{}/output/b2_to_b1.png'.format(args.savefolder), dpi=500)

        print(model.get_loss_names())
        print("{} ({}): {}".format(i, now(), lstring))



model.save(folder=args.savefolder)
print("Done! Final model saved.")






