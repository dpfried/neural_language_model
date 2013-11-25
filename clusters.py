from utils import models_in_folder, load_model
from query import query, get_vocab_container
import os
from pprint import pprint

def make_comparison_function(model_directories):
    model_paths = map(models_in_folder, model_directories)
    max_m = max(reduce(lambda p, q: p.intersection(q), map(lambda d: set(d.keys()), model_paths)))
    paths = [d[max_m] for d in model_paths]
    models = [load_model(path) for path in paths]
    common_prefix = os.path.commonprefix(paths)
    def suffix(model_path):
        if common_prefix != model_path:
            return os.path.relpath(model_path, common_prefix)
        else:
            return model_path

    vocab_container = get_vocab_container(models[0])
    def foo(word, n=10):
        for model, path in zip(models, paths):
            print suffix(path)
            pprint(query(model, vocab_container, word, n = n))
    return foo

if __name__ == "__main__":
    base = '/cl/work/dfried/models/adagrad/'
    distance_comparison = make_comparison_function([base + 'only_syntactic/no_adagrad', base + 'only_semantic/no_adagrad', base + 'no_init_0.01/no_adagrad'])
    translational_comparison = make_comparison_function([base + 'only_syntactic/no_adagrad', base + 'relational_only_semantic/no_adagrad', base + 'relational_no_init_0.01/no_adagrad'])
    tensor_comparison = make_comparison_function([base + 'only_syntactic/no_adagrad', base + 'tensor_only_semantic/no_adagrad', base + 'tensor_no_init_0.01/no_adagrad'])

