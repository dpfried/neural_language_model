import sys
import model
import relational
import relational.ntn
sys.modules['ntn'] = relational.ntn
sys.modules['model_new'] = model
sys.modules['wordnet_rels'] = relational.wordnet_rels
import theano

theano.tensor.basic.Subtensor = theano.tensor.Subtensor
