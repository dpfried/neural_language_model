import numpy as np
from model import NLM
import cPickle
import time
import sys
from ngrams import NgramReader
from utils import sample_cumulative_discrete_distribution
from wordnet_similarity import scaled_lch_similarity, pairwise_similarity, WunschPaths
from nltk.corpus import wordnet

def constant_weight(model, semantic_similarity_fn, replacement_column_index, correct_symbols, error_symbols, **kwargs):
    return 1

def semantic_vs_embedding_weight(model, semantic_similarity_fn, replacement_column_index, correct_symbols, error_symbols, **kwargs):
    """
    given the NLM (which contains word embeddings) and the index of the word that was corrupted
    to make the error_symbols from correct_symbols, return the degree by which the cost function
    should be weighted for this training example

    @type model: L{NLM}
    @param model: The instance of the NLM, containing the embeddings and vocabulary

    @type semantic_similarity_fn: a function L{Synset} x L{Synset} -> float
    @param semantic_similarity_fn: function of two synsets that returns a value between
    -1 and 1 reflecting their similarity, where -1 is least similar and 1 is
    most similar

    @type replacement_column_index: int
    @param replacement_column_index: The index of the word that was corrupted

    @type correct_symbols: list of int
    @param correct_symbols: the correct ngram as a list of indices into model.vocabulary

    @type error_symbols: list of int
    @param error_symbols: the error ngram as a list of indices into model.vocabulary

    @param kwargs: see wordnet_similarity.pairwise_similarity
    """
    correct_symbol = correct_symbols[replacement_column_index]
    error_symbol = error_symbols[replacement_column_index]

    if correct_symbol == 0 or error_symbol == 0:
        # we're dealing with the RARE word
        return 1

    # get the word embeddings corresponding to the target word in correct and
    # corrupted ngrams from the model
    correct_embedding = model.embedding_layer.embedding[correct_symbol]
    error_embedding = model.embedding_layer.embedding[error_symbol]

    # get the cosine similarity of the two word embeddings
    embedding_similarity = np.dot(correct_embedding, error_embedding) / (np.linalg.norm(correct_embedding, 2) * np.linalg.norm(error_embedding, 2))

    # get the lch_similarity from paths
    correct_word = model.vocabulary[correct_symbol]
    error_word = model.vocabulary[error_symbol]
    semantic_similarity = pairwise_similarity(wordnet.synsets(correct_word), wordnet.synsets(error_word), similarity_fn=semantic_similarity_fn, **kwargs)
    if semantic_similarity is not None:
        return np.exp(-1.0 * embedding_similarity * semantic_similarity)
    else:
        return 1

def test_nlm(vocab_size, dimensions, n_hidden, ngram_reader, rng=None, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0000, save_model_basename=None, blocks_to_run=np.inf, weight_fn=constant_weight, save_model_frequency=10, other_params={}):
    print '... building the model'

    if rng is None:
        rng = np.random.RandomState(1234)

    sequence_length = ngram_reader.ngram_length
    vocabulary = ngram_reader.word_array[:vocab_size]

    # load wordnet shortest paths object (this will take a while)
    # wordnet_shortest_paths = WunschPaths(wordnet.all_synsets())
    import cPickle, gzip
    with gzip.open('paths.pkl.gz', 'rb') as f:
        wordnet_shortest_paths = cPickle.load(f)

    # create a similarity function from this shortest path object
    scaled_similarity = lambda synset1, synset2: scaled_lch_similarity(wordnet_shortest_paths, synset1, synset2)

    nlm_model = NLM(rng=rng,
                    vocabulary=vocabulary,
                    dimensions=dimensions,
                    sequence_length=sequence_length,
                    n_hidden=n_hidden,
                    L1_reg=L1_reg,
                    L2_reg=L2_reg,
                    other_params=other_params)

    # replace the middle word in the ngram when producing corrupt examples
    replacement_column_index = sequence_length / 2

    print '... training'

    def process_data_row(row):
        """
        given a row of an ngram matrix reprsenting a training ngram, return a
        list of the symbols corresponding to words in that ngram, a corrupted
        list of the same symbols, and the frequency of the original ngram in
        the training corpus
        """
        # last two columns are reserved for frequency of ngram and cumulative
        # frequency, respectively
        correct_symbols = row[:-2]
        ngram_frequency = row[-2]
        # get a list of symbols representing a corrupted ngram
        # TODO: maybe move add_noise_to_symbols from ngram_reader to this file?
        error_symbols = ngram_reader.add_noise_to_symbols(correct_symbols, column_index=replacement_column_index, rng=rng)
        return correct_symbols, error_symbols, ngram_frequency


    last_time = time.clock()
    block_count = 0
    block_test_frequency = 1
    print_freq = 100

    # we will run through the entire testing block after each time we've
    # completed trained a training block
    testing_block = ngram_reader.testing_block()

    while block_count < blocks_to_run:
        block_count += 1
        costs = []
        training_block = ngram_reader.training_block(rng.random_sample())
        block_size = training_block.shape[0]
        # sample block_size ngrams from the training block, by frequency
        # in the original corpus. Using block_size as the sample size is
        # pretty arbitrary
        for count in xrange(block_size):
            if count % print_freq == 0:
                sys.stdout.write('\rblock %i: training instance %d of %d (%f %%)\r' % (block_count, count, block_size, 100. * count / block_size))
                sys.stdout.flush()
            train_index = sample_cumulative_discrete_distribution(training_block[:,-1])
            correct_symbols, error_symbols, ngram_frequency = process_data_row(training_block[train_index])
            # calculate the weight as a function of the correct symbols and error symbols
            weight = weight_fn(model=nlm_model,
                               semantic_similarity_fn=scaled_similarity,
                               replacement_column_index=replacement_column_index,
                               correct_symbols=correct_symbols,
                               error_symbols=error_symbols)
            costs.append(nlm_model.train(correct_symbols, error_symbols, learning_rate * weight))# * ngram_frequency))

        this_training_cost = np.mean(costs)
        # so that when we pickle the model we have a record of how many blocks
        # it's been trained on
        nlm_model.blocks_trained = block_count

        if block_count % block_test_frequency == 0:
            test_values = []
            test_frequencies = []
            n_test_instances = testing_block.shape[0]
            for test_index in xrange(n_test_instances):
                if test_index % print_freq == 0:
                    sys.stdout.write('\rtesting instance %d of %d (%f %%)\r' % (test_index, n_test_instances, 100. * test_index / n_test_instances))
                    sys.stdout.flush()
                correct_symbols, error_symbols, ngram_frequency = process_data_row(testing_block[test_index])
                test_values.append(nlm_model.score(correct_symbols) - nlm_model.score(error_symbols))
                test_frequencies.append(ngram_frequency)
            test_mean = np.mean(test_values)
            test_weighted_mean = np.mean(np.array(test_values) * np.array(test_frequencies))
            test_score_str = '%f' % test_mean
            test_wt_score_str = '%f' % test_weighted_mean
        else:
            test_score_str = '-----'
            test_wt_score_str = '-----'

        current_time = time.clock()
        sys.stdout.write('\033[k\r')
        sys.stdout.flush()
        print 'block %i \t training cost %f %% \t test score %s \t test wt score %s \t %f seconds' % (block_count, this_training_cost, test_score_str, test_wt_score_str, current_time - last_time)
        last_time = current_time

        if save_model_basename and block_count % save_model_frequency == 0:
            sys.stdout.write('dumping to file..\r')
            sys.stdout.flush()
            with gzip.open('%s-%d.pkl.gz' % (save_model_basename, block_count), 'wb') as f:
                cPickle.dump(nlm_model, f)
            sys.stdout.write('\033[k\r')
            sys.stdout.flush()

    return nlm_model

if __name__ == '__main__':
    import argparse
    import gzip
    weight_functions = {
        'constant': constant_weight,
        'sve': semantic_vs_embedding_weight,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngram_filename', help="hdf5 file to load ngrams from")
    parser.add_argument('--model_basename', help="basename to write model to")
    parser.add_argument('--vocab_size', type=int, help="number of top words to include", default=5000)
    parser.add_argument('--rng_seed', type=int, help="random number seed", default=1234)
    parser.add_argument('--dimensions', type=int, help="dimension of word representations", default=20)
    parser.add_argument('--n_hidden', type=int, help="number of hidden nodes", default=30)
    parser.add_argument('--L1_reg', type=float, help="L1 regularization constant", default=0.0)
    parser.add_argument('--L2_reg', type=float, help="L2 regularization constant", default=0.0)
    parser.add_argument('--learning_rate', type=float, help="L2 regularization constant", default=0.01)
    parser.add_argument('--train_proportion', type=float, help="percentage of data to use as training", default=0.95)
    parser.add_argument('--test_proportion', type=float, help="percentage of data to use as testing", default=None)
    parser.add_argument('--weight_fn', type=str, help="weight function to use (%s)" % ' or '.join(weight_functions.keys()), default='constant')
    parser.add_argument('--save_model_frequency', type=int, help="save model every nth iteration", default=10)
    args = parser.parse_args()

    ngram_reader = NgramReader(args.ngram_filename, vocab_size=args.vocab_size, train_proportion=args.train_proportion, test_proportion=args.test_proportion)
    print 'corpus contains %i ngrams' % (ngram_reader.number_of_ngrams)
    print 'constructing model...'
    rng = np.random.RandomState(args.rng_seed)
    params = {
        'rng':rng,
        'vocab_size':ngram_reader.vocab_size,
        'dimensions':args.dimensions,
        'n_hidden':args.n_hidden,
        'L1_reg':args.L1_reg,
        'L2_reg':args.L2_reg,
        'save_model_basename':args.model_basename,
        'learning_rate': args.learning_rate,
        'blocks_to_run':np.inf,
        'weight_fn': weight_functions[args.weight_fn],
        'save_model_frequency': args.save_model_frequency,
    }
    other_params = {
        'ngram_filename': args.ngram_filename,
    }
    print params
    params['ngram_reader'] = ngram_reader
    params['other_params'] = other_params
    model = test_nlm(**params)
