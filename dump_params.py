from os.path import join
from utils import models_in_folder
import json
import cPickle, gzip
from admm import ADMMModel, AnnealingADMMModel
from semantic_network import SemanticDistance, SemanticNet
from model import _default_word # required to unpickle because some old models have this function referenced within the clas
from joint import JointModel

def dump_params(params, dump_filename):
    with open(dump_filename, 'w') as f:
        json.dump(params, f)

def dump_params_from_directory(model_directory, dump_filename='params.json'):
    try:
        models = models_in_folder(model_directory)
        first_model = min(models)
        model_path = models[first_model]
        with gzip.open(model_path, 'rb') as f:
            model = cPickle.load(f)
    except StopIteration:
        print 'no models found in %s' % model_directory
        return
    except IOError as e:
        print 'IO error for %s' % model_directory
        print e
        return
    dump_params(model.other_params, join(model_directory, dump_filename))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_directories', nargs='+')
    args = parser.parse_args()

    for model_dir in args.model_directories:
        print 'dumping params for %s' % model_dir
        dump_params_from_directory(model_dir)
