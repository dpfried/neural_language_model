import numpy as np
from itertools import izip_longest
import re
import glob
import os

def sample_cumulative_discrete_distribution(cumulative_dist, rng=None):
    if rng is None:
        rng = np.random
    max_val = cumulative_dist[-1]
    sampled_val = rng.randint(max_val)
    return np.searchsorted(cumulative_dist, sampled_val)

def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') --> ('a', 'b', 'c'), ('d', 'e', 'f'), ('g', 'x', 'x')"""
    return izip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

MODEL_NUMBER_EXTRACTOR = re.compile(r'.*/model-(\d+).pkl.gz')
def models_in_folder(folder):
    get_model_fname = lambda model_num: os.path.join(folder, 'model-%s.pkl.gz' % model_num)
    model_fnames = glob.glob(get_model_fname('*'))
    return dict((int(MODEL_NUMBER_EXTRACTOR.match(fname).groups()[0]), fname)
                for fname in model_fnames)
