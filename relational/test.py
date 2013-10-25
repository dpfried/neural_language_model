import numpy as np
from relational.ntn import NeuralTensorNetwork
from model_new import NLM
from relational.relational_admm import RelationalADMMModel

N_DIM = 50
VOCAB_SIZE = 5
SEQ_LENGTH = 5
VOCAB = map(str, xrange(VOCAB_SIZE))

sem_model = NeuralTensorNetwork(np.random, VOCAB, 15, N_DIM, 100)
syn_model = NLM(np.random, VOCAB, N_DIM, SEQ_LENGTH, 100)
rel_model = RelationalADMMModel(syn_model, sem_model, VOCAB_SIZE , 0.01, {})

syn = np.arange(SEQ_LENGTH).astype('int32')
syn_corrupt = syn.copy()
syn_corrupt[2] += 1
syn_corrupt = np.mod(syn_corrupt, VOCAB_SIZE)
print 'update only syn'
syn_model.train(syn, syn_corrupt)

print 'update only sem'
sem_model.train(0,0,0,0,0,1)

print 'joint syntactic update'
rel_model.update_syntactic(syn, syn_corrupt)

print 'joint semantic update'
rel_model.update_semantic(0,0,0,0,0,1)

print 'update y'
rel_model.update_y()
