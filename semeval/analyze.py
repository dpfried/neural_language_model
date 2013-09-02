#!/usr/bin/env python
from glob import glob
import numpy as np

def attr_dict(filename):
    with open(filename) as f:
        keys_and_vals = [line.split(':') for line in f if line.strip()]
    return dict((key.strip(), val.strip()) for key, val in keys_and_vals)

def parse_correlation(filename):
    attrs = attr_dict(filename)
    return float(attrs['Spearman Correlation'])

def parse_accuracy(filename):
    attrs = attr_dict(filename)
    return float(attrs['Overall Accuracy'].strip('%')) / 100.

# compute average correlation
corrs = [parse_correlation(filename) for filename in glob('NLM/Testing/SpearmanNLMScaled-*.txt')]

# compute average accuracy
accuracy = [parse_accuracy(filename) for filename in glob('NLM/Testing/MaxDiffNLM-*.txt')]

print 'average correlation: %f' % np.mean(corrs)
print 'average accuracy: %f' % np.mean(accuracy)
