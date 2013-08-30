import numpy as np
from model import NLM
import cPickle
import time
import sys

def test_nlm(vocab_size, dimensions, n_hidden, data, rng=None, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0000, n_epochs=1000, save_model_basename=None, epochs=np.inf, other_params={}):
    print '... building the model'

    train_set_x, test_set_x = data

    n_train_instances, cols = train_set_x.shape
    n_test_instances, _ = test_set_x.shape

    sequence_length = cols - 1

    if rng is None:
        rng = np.random.RandomState(1234)

    nlm_model = NLM(rng=rng,
                    vocab_size=vocab_size,
                    dimensions=dimensions,
                    sequence_length=sequence_length,
                    n_hidden=n_hidden,
                    L1_reg=L1_reg,
                    L2_reg=L2_reg,
                    other_params=other_params)

    print '... training'

    def process_data_row(row):
        correct_symbols = row[:-1]
        ngram_frequency = row[-1]
        error_symbols = ngrams_reader.add_noise_to_symbols(correct_symbols, vocab_size=vocab_size, rng=rng)
        return correct_symbols, error_symbols, ngram_frequency

    last_time = time.clock()
    epoch = 0
    epoch_test_frequency = 1
    while epoch < epochs:
        epoch += 1
        print_freq = 100
        costs = []
        # for stochastic gradient descent, feed each training example in random order
        for count, train_index in enumerate(rng.permutation(n_train_instances)):
            if count % print_freq == 0:
                sys.stdout.write('\rtraining instance %d of %d (%f %%)\r' % (count, n_train_instances, 100. * count / n_train_instances))
                sys.stdout.flush()
            correct_symbols, error_symbols, ngram_frequency = process_data_row(train_set_x[train_index])
            costs.append(nlm_model.train(correct_symbols, error_symbols, learning_rate * ngram_frequency))

        this_training_cost = np.mean(costs)

        if epoch % epoch_test_frequency == 0:
            test_values = []
            test_frequencies = []
            for test_index in xrange(n_test_instances):
                if test_index % print_freq == 0:
                    sys.stdout.write('\rtesting instance %d of %d (%f %%)\r' % (test_index, n_test_instances, 100. * test_index / n_test_instances))
                    sys.stdout.flush()
                correct_symbols, error_symbols, ngram_frequency = process_data_row(test_set_x[test_index])
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
        print 'epoch %i \t training cost %f %% \t test score %s \t test wt score %s \t %f seconds' % (epoch, this_training_cost, test_score_str, test_wt_score_str, current_time - last_time)
        last_time = current_time

        if save_model_basename:
            sys.stdout.write('dumping to file..\r')
            sys.stdout.flush()
            with gzip.open('%s-%d.pkl.gz' % (save_model_basename, epoch), 'wb') as f:
                cPickle.dump(nlm_model, f)
            sys.stdout.write('\033[k\r')
            sys.stdout.flush()

    return nlm_model

if __name__ == '__main__':
    import argparse
    import ngrams_reader
    import gzip
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_ngrams', help="path to write ngram pickle to")
    parser.add_argument('--load_ngrams', help="path to load ngram pickle from (supersedes")
    parser.add_argument('--model_basename', help="basename to write model to")
    parser.add_argument('--corpus_size', type=int, help="0, 1, 2, or 3", default=0)
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

    ngram_filename = args.write_ngrams or args.load_ngrams

    if args.load_ngrams:
        print 'loading n_grams from %s...' % args.load_ngrams
        if args.corpus_size or args.vocab_size:
            print "warning: ignoring corpus_size and vocab_size, since we're loading the corpus from file"
        with gzip.open(args.load_ngrams, 'rb') as f:
            n_grams = cPickle.load(f)
    else:
        print 'building ngrams from corpus...'
        n_grams = ngrams_reader.NGramsContainer(ngrams_reader.DATA_BY_SIZE[args.corpus_size], num_words=args.vocab_size)
    print 'corpus contains %i ngrams, with %i word types. vocabulary is %i types' % (n_grams.n_examples, len(n_grams.frequency_counts), n_grams.vocab_size)
    if args.write_ngrams:
        print 'dumping n_grams to %s...' % args.write_ngrams
        with gzip.open(args.write_ngrams, 'wb') as f:
            cPickle.dump(n_grams, f)

    print 'extracting data...'
    rng = np.random.RandomState(args.rng_seed)
    data = n_grams.get_data(rng=rng, train_proportion=args.train_proportion, test_proportion=args.test_proportion)
    print 'constructing model...'
    params = {
        'rng':rng,
        'vocab_size':n_grams.vocab_size,
        'dimensions':args.dimensions,
        'n_hidden':args.n_hidden,
        'L1_reg':args.L1_reg,
        'L2_reg':args.L2_reg,
        'save_model_basename':args.model_basename,
        'learning_rate': args.learning_rate,
        'epochs':np.inf,
    }
    other_params = {
        'ngram_filename': ngram_filename,
    }
    print params
    params['data'] = data
    params['other_params'] = other_params
    model = test_nlm(**params)
