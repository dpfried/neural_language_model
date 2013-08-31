import numpy as np

def sample_cumulative_discrete_distribution(cumulative_dist, rng=None):
    if rng is None:
        rng = np.random
    max_val = cumulative_dist[-1]
    sampled_val = rng.randint(max_val)
    return np.searchsorted(cumulative_dist, sampled_val)
