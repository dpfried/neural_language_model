# coding: utf-8
import gzip, cPickle
from relational_admm import *
theano.tensor.basic.Subtensor = theano.tensor.Subtensor # for old models pickled with theano 0.5

from config import DEFAULT_NGRAM_FILENAME
def load_vocab(model):
    model._load_vocab(DEFAULT_NGRAM_FILENAME) # since we don't save the vocab in many semantic models

with gzip.open('/cl/work/dfried/models/relational/admm_0.01/model-650.pkl.gz') as f:
    model = cPickle.load(f)
    load_vocab(model.semantic_model)

with gzip.open('/cl/work/dfried/models/factorial/no_init_0.01/model-650.pkl.gz') as f:
    model_dist = cPickle.load(f)
    load_vocab(model_dist.semantic_model)

from query import query
print query(model, 'king')
print query(model.syntactic_model, 'king')
print query(model.semantic_model, 'king')

print query(model_dist, 'king')
print query(model_dist.syntactic_model, 'king')
print query(model_dist.semantic_model, 'king')

