import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def make_legend(ax, labels, s=20, cmap=mpl.cm.jet, **kwargs):
    uniquelabs = np.unique(labels)
    numlabs = len(uniquelabs)
    for i, label in enumerate(uniquelabs):
        if numlabs > 1:
            ax.scatter(0, 0, s=s, c=[cmap(1 * i / (numlabs - 1))], label=label)
        else:
            ax.scatter(0, 0, s=s, c=[cmap(1.)], label=label)
    ax.scatter(0, 0, s=2 * s, c='w')
    ax.legend(**kwargs)

def truncate_colormap(cmap, vmin=0.0, vmax=1.0, n=100):
    str_ = 'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=vmin, b=vmax)
    cmap_ = cmap(np.linspace(vmin, vmax, n))
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(str_, cmap_)
    return new_cmap
