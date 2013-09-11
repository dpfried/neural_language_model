import numpy as np
from itertools import izip_longest

def sample_cumulative_discrete_distribution(cumulative_dist, rng=None):
    if rng is None:
        rng = np.random
    max_val = cumulative_dist[-1]
    sampled_val = rng.randint(max_val)
    return np.searchsorted(cumulative_dist, sampled_val)

def grouper(n, iterable, padvalue=None):
    """grouper(3, 'abcdefg', 'x') --> ('a', 'b', 'c'), ('d', 'e', 'f'), ('g', 'x', 'x')"""
    return izip_longest(*[iter(iterable)]*n, fillvalue=padvalue)
