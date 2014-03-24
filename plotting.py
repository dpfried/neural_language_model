import numpy as np
import matplotlib.pyplot as plt

def barplot_series(multiple_series, names, ylabels=None, title=None, xlabel=None, xrange=None):
    N = len(multiple_series[0])
    if not(all(len(xs) == N for xs in multiple_series)):
        raise ValueError("not all series are same length")
    ind = np.arange(N)

    height = 0.35

    fig, ax = plt.subplots()
    rectss = []
    for xs in multiple_series:
        rects = ax.barh(ind, xs, height)
        rectss.append(rects)

    ax.legend(rectss, names)

    if ylabels:
        ax.set_yticks(ind+height)
        ax.set_yticklabels(ylabels)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if xrange:
        ax.set_xrange(xrange)
