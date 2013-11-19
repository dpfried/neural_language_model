import cPickle, gzip
from admm import *
from model import *
from semantic_network import *
import gzip
import cPickle

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='zipped model file to read embeddings from')
    parser.add_argument('dump_filename', help='filename to dump the embeddings to')
    args = parser.parse_args()

    with gzip.open(args.model) as f:
        model = cPickle.load(f)

    model.dump_embeddings(args.dump_filename)
