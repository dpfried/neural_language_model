import numpy as np
from model import NLM
import time
import sys
from ngrams import NgramReader
from utils import sample_cumulative_discrete_distribution
from nltk.corpus import wordnet
import pandas
import cPickle

def test_nlm(vocab_size, dimensions, synset_dimensions, n_hidden, ngram_reader, rng=None, synset_sampling_rng=None, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0000, save_model_basename=None, blocks_to_run=np.inf, save_model_frequency=10, other_params={}, stats_output_file=None, update_related_weight=1.0, take_top_synset=False):
    print '... building the model'

    if rng is None:
        rng = np.random.RandomState(1234)

    if synset_sampling_rng is None:
        synset_sampling_rng = np.random.RandomState(1234)

    sequence_length = ngram_reader.ngram_length
    vocabulary = ngram_reader.word_array[:vocab_size]

    id_to_word = dict(enumerate(vocabulary))
    word_to_id = dict((word, index) for (index, word) in enumerate(vocabulary))

    synsets = ['NONE'] + list(wordnet.all_synsets())

    id_to_synset = dict(enumerate(synsets))
    synset_to_id = dict((synset, index) for (index, synset) in enumerate(synsets))

    def sample_synset(word_symbol):
        possible_synsets = wordnet.synsets(id_to_word[word_symbol])
        if not possible_synsets:
            return 0
        else:
            if take_top_synset:
                choice = 0
            else:
                choice = synset_sampling_rng.randint(len(possible_synsets))
            return synset_to_id[possible_synsets[choice]]

    def sample_synsets(word_symbol_sequence):
        """given an ngram of word indices, uniformly sample a synset for each
        word and then return a list of the indices of these symbols"""
        return [sample_synset(word_symbol) for word_symbol in word_symbol_sequence]

    nlm_model = NLM(rng=rng,
                    vocabulary=vocabulary,
                    synsets=synsets,
                    dimensions=dimensions,
                    synset_dimensions=synset_dimensions,
                    sequence_length=sequence_length,
                    n_hidden=n_hidden,
                    L1_reg=L1_reg,
                    L2_reg=L2_reg,
                    other_params=other_params)

    # replace the middle word in the ngram when producing corrupt examples
    replacement_column_index = sequence_length / 2

    all_stats = pandas.DataFrame()

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
        stats_for_block = {}
        for count in xrange(block_size):
            if count % print_freq == 0:
                sys.stdout.write('\rblock %i: training instance %d of %d (%f %%)\r' % (block_count, count, block_size, 100. * count / block_size))
                sys.stdout.flush()
            train_index = sample_cumulative_discrete_distribution(training_block[:,-1])
            correct_symbols, error_symbols, ngram_frequency = process_data_row(training_block[train_index])
            correct_synset_symbols = sample_synsets(correct_symbols)
            error_synset_symbols = sample_synsets(error_symbols)
            # calculate the weight as a function of the correct symbols and error symbols
            cost, correct_updates, error_updates, correct_synset_updates, error_synset_updates = nlm_model.train(correct_symbols, error_symbols, correct_synset_symbols, error_synset_symbols, learning_rate) # * ngram_frequency
            costs.append(cost)

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
                correct_synset_symbols = sample_synsets(correct_symbols)
                error_synset_symbols = sample_synsets(error_symbols)
                test_values.append(nlm_model.score(correct_symbols, correct_synset_symbols) - nlm_model.score(error_symbols, error_synset_symbols))
                test_frequencies.append(ngram_frequency)
            test_mean = np.mean(test_values)
            stats_for_block['test_mean'] = test_mean
            test_weighted_mean = np.mean(np.array(test_values) * np.array(test_frequencies))
            stats_for_block['test_weighted_mean'] = test_weighted_mean
            test_score_str = '%f' % test_mean
            test_wt_score_str = '%f' % test_weighted_mean
        else:
            test_score_str = '-----'
            test_wt_score_str = '-----'

        current_time = time.clock()
        stats_for_block['time'] = current_time - last_time
        stats_for_block['training_cost'] = this_training_cost
        sys.stdout.write('\033[k\r')
        sys.stdout.flush()
        print 'block %i \t training cost %f %% \t test score %s \t test wt score %s \t %f seconds' % (block_count, this_training_cost, test_score_str, test_wt_score_str, current_time - last_time)
        last_time = current_time

        all_stats = pandas.concat([all_stats, pandas.DataFrame(stats_for_block, index=[block_count])])

        if save_model_basename and block_count % save_model_frequency == 0:
            sys.stdout.write('dumping to file..\r')
            sys.stdout.flush()
            with gzip.open('%s-%d.pkl.gz' % (save_model_basename, block_count), 'wb') as f:
                cPickle.dump(nlm_model, f)
            sys.stdout.write('\033[k\r')
            sys.stdout.flush()

        if stats_output_file:
            print 'dumping stats to file %s' % stats_output_file
            all_stats.to_pickle(stats_output_file)

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
    parser.add_argument('--synset_dimensions', type=int, help="dimension of synset representations", default=20)
    parser.add_argument('--n_hidden', type=int, help="number of hidden nodes", default=30)
    parser.add_argument('--L1_reg', type=float, help="L1 regularization constant", default=0.0)
    parser.add_argument('--L2_reg', type=float, help="L2 regularization constant", default=0.0)
    parser.add_argument('--learning_rate', type=float, help="L2 regularization constant", default=0.01)
    parser.add_argument('--train_proportion', type=float, help="percentage of data to use as training", default=0.95)
    parser.add_argument('--test_proportion', type=float, help="percentage of data to use as testing", default=None)
    parser.add_argument('--take_top_synset', action='store_true', help="don't sample synsets, choose the most likely")
    parser.add_argument('--save_model_frequency', type=int, help="save model every nth iteration", default=10)
    parser.add_argument('--stats_output_file', type=str, help="save stats to this file")
    parser.add_argument('--update_related_weight', type=float, help="scale of updates to words in same synset as replaced and replacement words")
    args = parser.parse_args()

    ngram_reader = NgramReader(args.ngram_filename, vocab_size=args.vocab_size, train_proportion=args.train_proportion, test_proportion=args.test_proportion)
    print 'corpus contains %i ngrams' % (ngram_reader.number_of_ngrams)
    print 'constructing model...'
    rng = np.random.RandomState(args.rng_seed)
    params = {
        'rng':rng,
        'vocab_size':ngram_reader.vocab_size,
        'dimensions':args.dimensions,
        'synset_dimensions':args.synset_dimensions,
        'n_hidden':args.n_hidden,
        'L1_reg':args.L1_reg,
        'L2_reg':args.L2_reg,
        'save_model_basename':args.model_basename,
        'learning_rate': args.learning_rate,
        'blocks_to_run':np.inf,
        'stats_output_file': args.stats_output_file,
        'save_model_frequency': args.save_model_frequency,
        'update_related_weight': args.update_related_weight,
        'take_top_synset': args.take_top_synset,
    }
    other_params = {
        'ngram_filename': args.ngram_filename,
    }
    print params
    params['ngram_reader'] = ngram_reader
    params['other_params'] = other_params
    model = test_nlm(**params)
