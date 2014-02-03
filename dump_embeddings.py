from utils import load_model
import gzip
from os.path import join
import ngrams
from query import DEFAULT_NGRAM_FILENAME

def get_vocab(model):
    try:
        base_dir = model.other_params['base_dir']
        vocabulary_path = join(base_dir, 'vocabulary.pkl.gz')
        return load_model(vocabulary_path)
    except:
        try:
            ngram_filename = model.other_params['ngram_filename']
        except:
            ngram_filename = DEFAULT_NGRAM_FILE
        try:
            vocab_size = model.other_params['vocab_size']
        except:
            vocab_size = 50000
        return ngrams.NgramReader(ngram_filename, vocab_size=vocab_size).word_array


if __name__ == "__main__":
    import config
    config.DYNAMIC["compile_on_load"] = False

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='zipped model file to read embeddings from')
    parser.add_argument('dump_filename', help='filename to dump the embeddings to')
    args = parser.parse_args()

    model = load_model(args.model)
    vocab = get_vocab(model)

    with gzip.open(args.dump_filename, 'w') as f:
        for word, row in zip(vocab, model.averaged_embeddings()):
            f.write("%s %s\n" % (word, ' '.join(map(str, row))))
