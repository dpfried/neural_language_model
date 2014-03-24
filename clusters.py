from utils import models_in_folder, load_model, map_dict
from query import query
from pprint import pprint
import ngrams
from os.path import join
from config import DEFAULT_NGRAM_FILENAME

from query impmort get_vocab

def make_comparison_function(model_directories, model_number=None):
    model_paths = map_dict(models_in_folder, model_directories)
    if model_number is None:
        model_number = max(reduce(lambda p, q: p.intersection(q), map(lambda d: set(d.keys()), model_paths.values())))

    paths = map_dict(lambda d: (d[model_number] if d else None), model_paths)
    models = map_dict(load_model, paths)

    vocab_containers = map_dict(get_vocab, models)
    # since we turned off compile on load...
    for v in vocab_containers.values():
        try:
            v._initialize()
        except:
            pass
    def display_cluster(word, n=10):
        for name, model in models.iteritems():
            print "-------"
            print name
            pprint(query(model, vocab_containers[name], word, n = n))
            print
    return display_cluster, model_number


if __name__ == "__main__":
    models = {
        'GD+NLM': '/cl/work/dfried/models/socher_dataset_1-12/gd/weight0.5/',
        'NTN+NLM': '/cl/work/dfried/models/socher_dataset_1-12/ntn/weight0.5/',
        'TransE+NLM': '/cl/work/dfried/models/socher_dataset_1-12/transe/weight0.5/',
        'NLM': '/cl/work/dfried/models/adagrad/only_syntactic/no_adagrad/',
        'GD': '/cl/work/dfried/models/adagrad/only_semantic/no_adagrad/',
        'NTN': '/cl/work/dfried/models/socher_dataset_1-10/ntn/only_sem/',
        'TransE': '/cl/work/dfried/models/socher_dataset_1-10/transe/only_sem/'
    }

    import config
    config.DYNAMIC['compile_on_load'] = False

    print "loading models.."
    print_cluster, _ = make_comparison_function(models, model_number=1000)

    while True:
        query_word = raw_input('enter word to query or X to stop:')
        if query_word == 'X':
            break
        print_cluster(query_word)
