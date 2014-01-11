from nltk.corpus import wordnet as wn
import random
from functional import mapcan
import numpy as np
import os

from collections import defaultdict
from vocab import Vocabulary, UNKNOWN_WORD

import nltk


# names of L{Synset} methods that return other synsets
relationship_methods = ['hypernyms', 'instance_hypernyms', 'hyponyms', 'instance_hyponyms',
                        'member_holonyms', 'substance_holonyms', 'part_holonyms',
                        'member_meronyms', 'substance_meronyms', 'part_meronyms',
                        'attributes', 'entailments', 'causes', 'also_sees', 'verb_groups',
                        'similar_tos']

DEFAULT_PATH = '/home/dfried/data/codeDeepDB/data/Wordnet/'
def train_path(path=DEFAULT_PATH):
    return os.path.join(path, 'train.txt')
def dev_path(path=DEFAULT_PATH):
    return os.path.join(path, 'dev.txt')
def test_path(path=DEFAULT_PATH):
    return os.path.join(path, 'test.txt')

## operates on the Wordnet relation dataset available at Richard Socher's website, http://www.socher.org/index.php/Main/ReasoningWithNeuralTensorNetworksForKnowledgeBaseCompletion
## we need to map the entity tokens in the dataset to NLTK Synsets, and the relationship tokens in the dataset to Synset method names

# a manually culled dictionary of the type produced by resolve_relationships,
# can do some experiments by passing this as the 2nd arg to resolve_synsets
thresholded_rel_map = {
    '_has_instance': ['hyponyms', 'instance_hyponyms'],
    '_similar_to': ['similar_tos'],
    '_member_meronym': ['member_meronyms', 'part_meronyms'],
    '_domain_region': ['hyponyms', 'part_meronyms'],
    '_subordinate_instance_of': ['instance_hypernyms'],
    '_domain_topic': ['part_meronyms', 'member_holonyms'],
    '_member_holonym': ['member_holonyms'],
    '_synset_domain_topic': ['hypernyms', 'part_holonyms'],
    '_has_part': ['part_meronyms'],
    '_part_of': ['part_holonyms', 'substance_holonyms'],
    '_type_of': ['hypernyms'],
}

def parse_synset_token(name):
    '''__spiritual_bouquet_1 -> spiritual_bouquet, 1'''
    tokens = name.split('_')[2:]
    return '_'.join(tokens[:-1]), int(tokens[-1])

def read_lines(filename):
    with open(filename) as f:
        return list(map(lambda s: s.rstrip().split(), f))

# def resolve_file(filename=default_filename):
#     '''
#     Run entity and relationship resolution on the Socher et al training dataset, returning one to many mapping dictionaries.
#     Warning: if a relation or synset can't be resolved, it won't be present in the dictionary, so these are not total functions

#     @return lines: a list of [entity_a_token, relation_token, entity_b_token] lists
#     @return rel_map: maps relation tokens from the dataset to sets of NLTK Synset method names
#     @return syn_map: maps entity tokens from the dataset to sets of NLTK Synsets
#     '''
#     lines = read_lines(filename)
#     rel_map = resolve_relationships(lines)
#     return lines, rel_map, resolve_synsets(lines, rel_map)

def parse_lines(lines):
    def parse_line(line):
        token_a, relation, token_b = line[:3]
        if len(line) == 4:
            label = int(line[3])
        else:
            label = 1
        return token_a, relation, token_b, label
    return map(parse_line, lines)

def resolve_relationships(lines):
    rel_map = defaultdict(nltk.FreqDist)
    relationship_types = set()
    for token_a, relation, token_b, label in lines:
        if label == -1:
            continue
        assert label == 1
        relationship_types.add(relation)
        synset_name_a, _ = parse_synset_token(token_a)
        synset_name_b, _ = parse_synset_token(token_b)
        possible_synsets_a = set(wn.synsets(synset_name_a))
        possible_synsets_b = set(wn.synsets(synset_name_b))

        # attempt resolution
        possible_matches = set()
        for syn_a in possible_synsets_a:
            for rel in relationship_methods:
                tails = set(syn_a.__getattribute__(rel)())
                if tails & possible_synsets_b:
                    possible_matches.add(rel)
        if possible_matches and len(possible_matches) == 1:
            rel = iter(possible_matches).next()
            rel_map[relation].inc(rel)
    return rel_map

def resolve_synsets(lines, rel_map):
    synset_map = defaultdict(nltk.FreqDist)
    synset_types = set()
    for token_a, relation, token_b, label in lines:
        if label == -1:
            continue
        assert label == 1
        synset_types.add(token_a)
        synset_types.add(token_b)
        synset_name_a, _ = parse_synset_token(token_a)
        synset_name_b, _ = parse_synset_token(token_b)
        possible_synsets_a = set(wn.synsets(synset_name_a))
        possible_synsets_b = set(wn.synsets(synset_name_b))

        # attempt resolution
        possible_matches = []
        for syn_a in possible_synsets_a:
            for rel in rel_map[relation]:
                tails = set(syn_a.__getattribute__(rel)())
                for syn_b in (tails & possible_synsets_b):
                    possible_matches.append((syn_a, rel, syn_b))
        if possible_matches and len(possible_matches) == 1:
            syn_a, rel, syn_b = possible_matches[0]
            synset_map[token_a].inc(syn_a)
            synset_map[token_b].inc(syn_b)
    print 'num synsets: %d' % len(synset_types)
    print 'synsets resolved: %d' % len(synset_map)
    return synset_map

class RelationshipsNTNDataset(object):
    def __init__(self, existing_vocab=None, rng=np.random, dataset_path=DEFAULT_PATH):
        '''existing_vocab should have the UNKNOWN_WORD represented by position 0'''
        self.rng = rng

        # read in the data file
        self.train_raw = parse_lines(read_lines(train_path(dataset_path)))
        self.dev_raw = parse_lines(read_lines(dev_path(dataset_path)))
        self.test_raw = parse_lines(read_lines(test_path(dataset_path)))


        # resolve relationships for use in resolving synsets
        rel_map = resolve_relationships(self.train_raw + self.dev_raw + self.test_raw)

        # maps tokens of the form __dog_1 to a FreqDist where keys are possible
        # synsets
        self.syn_map = resolve_synsets(self.train_raw + self.dev_raw + self.test_raw,
                                       rel_map)
        print 'collecting synsets'
        self.synsets = set(syn for fd in self.syn_map.values()
                           for syn in fd)


        print 'collecting words'
        # figure out what words are in these synsets
        synset_lemma_words = set(word for syn in self.synsets
                                 for word in words_from_synset(syn))
        if existing_vocab is None:
            existing_vocab = [UNKNOWN_WORD]

        existing_vocab_set = set(existing_vocab)

        print 'finding new words'
        # what words in the synsets aren't already in the vocab?
        new_words = synset_lemma_words - existing_vocab_set
        # don't forget about rare word, this should be 0 in existing vocab
        print 'creating vocabulary'
        self.vocabulary = Vocabulary(list(existing_vocab) + list(new_words))
        print 'mapping synsets to words'
        self.synset_to_word = SynsetToWord(self.vocabulary, self.synsets)

        self.relations = list(rel_map.keys())

        def words(synset_token):
            return list(set( word for synset in self.syn_map[synset_token]
                    for word in self.synset_to_word.words_by_synset[synset] ))
        def relation(rel_token):
            return self.relations.index(rel_token)

        def make_words_set(raw_lines):
            return [(words(syn_a), relation(rel), words(syn_b), label)
                    for syn_a, rel, syn_b, label in raw_lines]

        # each is a list of
        # ([word indices in synset_a], rel_name, [words indices in synset b], label)
        # tuples
        print 'making train set'
        self.train_words = make_words_set(self.train_raw)
        print 'making dev set'
        self.dev_words = make_words_set(self.dev_raw)
        print 'making test set'
        self.test_words = make_words_set(self.test_raw)

        self.N_relationships = len(self.relations)
        self.N_train = len(self.train_words)
        self.N_dev = len(self.dev_words)
        self.N_test = len(self.test_words)

        self.indices_in_intersection = {self.vocabulary[i] for i in (existing_vocab_set & synset_lemma_words)}
        # for indices_a, _, indices_b, _ in self.train_words:
        #     self.indices_in_intersection.update(indices_a)
        #     self.indices_in_intersection.update(indices_b)

    def training_block(self):
        for i in self.rng.permutation(self.N_train):
            index_list_a, index_rel, index_list_b, _ = self.train_words[i]
            if index_list_a and index_list_b:
                # sample synsets
                index_a = self.rng.choice(index_list_a)
                index_b = self.rng.choice(index_list_b)
                yield index_a, index_rel, index_b

    def display_word_row(self, row):
        indices_a, rel_index, indices_b, label = row
        def lookup_words(indices):
            return [self.vocabulary[index] for index in indices]
        return lookup_words(indices_a), self.relations[rel_index], lookup_words(indices_b), label

    def usable_row(self, row):
        indices_a, _, indices_b, _ = row
        return indices_a != [] and indices_b != []

class Relationships(object):
    relationships = relationship_methods

    def __init__(self, seed=1234):
        self.synsets = list(wn.all_synsets())
        self.indices = dict((syn, i) for i, syn in enumerate(self.synsets))
        self.seed = seed
        self.data = self._make_dataset(seed)
        self.N = len(self.data)

    def _make_dataset(self, seed):
        """ create the list of tuples and shuffle it by seed """
        rels = list(self.tuples(wn.all_synsets()))
        random.seed(seed)
        random.shuffle(rels)
        return np.array(rels)

    def indices_to_symbolic(self, indices_tuple):
        """ convert a tuple of indices (i.e. the type stored in self.data)
        into a tuple of Synset, string, Synset"""
        a, b, rel = indices_tuple
        return self.synsets[a], self.synsets[b], self.relationships[rel]

    def symbolic_to_indices(self, symbolic_tuple):
        """ convert a tuple of Synset, string, Synset
        into a tuple of indices (i.e. the type stored in self.data)
        """
        a, b, rel = symbolic_tuple
        return self.indices[a],  self.indices[b], self.relationships.index[rel]

    def tuples_for_synset(self, synset):
        """
        Return a list of tuples (synset_index, relation_index, other_synset_index),
        where synset_index is the index of this synset
        """
        synset_index = self.indices[synset]
        return ((synset_index, self.indices[other_synset], relation_index)
                for relation_index, method_name in enumerate(self.relationships) # go through all relation methods
                for other_synset in getattr(synset, method_name)()) # and call the method, getting a list of synsets in that relationship

    def tuples(self, synsets):
        """return a list of index tuples for all relationships which all of the given
        synsets participate in"""
        return mapcan(self.tuples_for_synset, synsets)

def words_from_synset(synset):
    return list(set(word.lower()
                    for lemma_name in synset.lemma_names
                    for word in lemma_name.split('_')))

class SynsetToWord(object):
    def __init__(self, vocabulary, synsets=None):
        '''vocabulary: L{Vocabulary}'''
        if synsets is None:
            synsets = wn.all_synsets()
        self.synsets = synsets
        self.vocabulary = vocabulary
        # self.word_index_to_token = vocabulary
        # self.word_token_to_index = dict((word, index)
        #                            for index, word in enumerate(vocabulary))
        self.words_by_synset = {
            synset: [self.vocabulary.symbol_to_index[word]
                     for word in words_from_synset(synset)
                     if word in self.vocabulary]
            for synset in self.synsets
        }

    # def all_words_in_relations(self, rels):
    #     """
    #     rels: wordnet_rels.Relationships
    #     """
    #     words = set()
    #     for row in rels.data:
    #         syn_a, syn_b, rel = rels.indices_to_symbolic(row)
    #         words.update(self.words_by_synset[syn_a])
    #         words.update(self.words_by_synset[syn_b])
    #     return words

    def usable(self, syn):
        return bool(self.words_by_synset[syn])

if __name__ == "__main__":
    rels = Relationships()
    import ngrams
    reader = ngrams.NgramReader('/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5', vocab_size=50000)
    s2w = SynsetToWord(reader.word_array)
    # indices_in_relationships = s3w.all_words_in_relations(rels)
    # print '%d words in vocabulary covered by relationships (out of %d)' % (len(indices_in_relationships) , len(reader.word_array))
