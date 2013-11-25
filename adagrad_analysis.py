from models import EmbeddingLayer
import theano
import theano.tensor as T
from utils import map_dict
from functools import partial

# def make_update(pooled=False):
#     learning_rate = T.scalar()
#     if pooled:
#         input = T.vector()
#         return theano.function(inputs=[learning_rate, input], outputs= learning_rate / input)
#     else:
#         input = T.matrix()
#         return theano.function(inputs=[learning_rate, input], outputs=learning_rate / T.sqrt(input))

def prefix_keys(dictionary, prefix):
    return {
        (prefix + '_' + key) : value
        for key, value in dictionary.iteritems()
    }

def component_updates(component, name=''):
    update_fn = lambda lr, w: lr.get_value() / w.get_value()
    d = map_dict(partial(update_fn, component.learning_rate), component.ada_weights)
    if name:
        return prefix_keys(d, name)
    else:
        return d

def model_updates(model, name=''):
    if not hasattr(model, 'components'):
        d = component_updates(model, name)
    else:
        d = {}
        for d_inc in [model_updates(getattr(model, component_name), component_name)
                      for component_name in model.components]:
            d.update(d_inc)
    if name:
        return prefix_keys(d, name)
    else:
        return d
