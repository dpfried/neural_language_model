import numpy as np
from model import NLM
import cPickle
import time
import sys
from ngrams import NgramReader
from utils import sample_cumulative_discrete_distribution

def test_nlm(vocab_size, dimensions, n_hidden, ngram_reader, rng=None, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0000, save_model_basename=None, blocks_to_run=np.inf, other_params={}):
    print '... building the model'

    if rng is None:
        rng = np.random.RandomState(1234)

    nlm_model = NLM(rng=rng,
                    vocab_size=vocab_size,
                    dimensions=dimensions,
                    sequence_length=ngram_reader.ngram_length,
                    n_hidden=n_hidden,
                    L1_reg=L1_reg,
                    L2_reg=L2_reg,
                    other_params=other_params)

    print '... training'

    def process_data_row(row):
        correct_symbols = row[:-2]
        ngram_frequency = row[-2]
        error_symbols = ngram_reader.add_noise_to_symbols(correct_symbols, rng=rng)
        return correct_symbols, error_symbols, ngram_frequency

    last_time = time.clock()
    block_count = 0
    block_test_frequency = 1
    print_freq = 100

    testing_block = ngram_reader.testing_block()

    while block_count < blocks_to_run:
        block_count += 1
        costs = []
        training_block = ngram_reader.training_block(rng.random_sample())
        block_size = training_block.shape[0]
        for count in xrange(block_size):
            if count % print_freq == 0:
                sys.stdout.write('\rblock %i: training instance %d of %d (%f %%)\r' % (block_count, count, block_size, 100. * count / block_size))
                sys.stdout.flush()
            train_index = sample_cumulative_discrete_distribution(training_block[:,-1])
            correct_symbols, error_symbols, ngram_frequency = process_data_row(training_block[train_index])
            costs.append(nlm_model.train(correct_symbols, error_symbols, learning_rate))# * ngram_frequency))

        this_training_cost = np.mean(costs)

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

        if save_model_basename:
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
    }
    other_params = {
        'ngram_filename': args.ngram_filename,
    }
    print params
    params['ngram_reader'] = ngram_reader
    params['other_params'] = other_params
    model = test_nlm(**params)
