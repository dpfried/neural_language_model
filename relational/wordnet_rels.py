from nltk.corpus import wordnet as wn
import random
from functional import mapcan
import numpy as np

from collections import defaultdict

import nltk


# names of L{Synset} methods that return other synsets
relationship_methods = ['hypernyms', 'instance_hypernyms', 'hyponyms', 'instance_hyponyms',
                        'member_holonyms', 'substance_holonyms', 'part_holonyms',
                        'member_meronyms', 'substance_meronyms', 'part_meronyms',
                        'attributes', 'entailments', 'causes', 'also_sees', 'verb_groups',
                        'similar_tos']

default_filename = '/home/dfried/data/codeDeepDB/data/Wordnet/train.txt'

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

def read_lines(filename=default_filename):
    with open(filename) as f:
        return list(map(lambda s: s.rstrip().split(), f))

def resolve_file(filename=default_filename):
    '''
    Run entity and relationship resolution on the Socher et al training dataset, returning one to many mapping dictionaries.
    Warning: if a relation or synset can't be resolved, it won't be present in the dictionary, so these are not total functions

    @return lines: a list of [entity_a_token, relation_token, entity_b_token] lists
    @return rel_map: maps relation tokens from the dataset to sets of NLTK Synset method names
    @return syn_map: maps entity tokens from the dataset to sets of NLTK Synsets
    '''
    lines = read_lines(filename)
    rel_map = resolve_relationships(lines)
    return lines, rel_map, resolve_synsets(lines, rel_map)

def resolve_relationships(lines):
    rel_map = defaultdict(nltk.FreqDist)
    relationship_types = set()
    for token_a, relation, token_b in lines:
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
    for token_a, relation, token_b in lines:
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
    def __init__(self, vocab, rng, filename=default_filename):
        self.synsets = list(wn.all_synsets())
        self.vocab = vocab
        self.rng = rng
        self.lines, self.rel_map, self.syn_map = resolve_file(filename)
        self.synset_to_word = SynsetToWord(vocab)

        self.relations = list(self.rel_map.keys())

        def words(synset_token):
            return [word for synset in self.syn_map[synset_token]
                    for word in self.synset_to_word.words_by_synset[synset]]
        def relation(rel_token):
            return self.relations.index(rel_token)

        self.data = [(words(syn_a), relation(rel), words(syn_b))
                      for syn_a, rel, syn_b in self.lines]
        self.N = len(self.data)

        self.N_relationships = len(self.relations)

        self.indices_in_intersection = set()
        for indices_a, _, indices_b in self.data:
            self.indices_in_intersection.update(indices_a)
            self.indices_in_intersection.update(indices_b)

    def training_block(self):
        for i in self.rng.permutation(self.N):
            index_list_a, index_rel, index_list_b = self.data[i]
            if index_list_a and index_list_b:
                # sample synsets
                index_a = self.rng.choice(index_list_a)
                index_b = self.rng.choice(index_list_b)
                # print self.lines[i]
                yield index_a, index_rel, index_b

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

class SynsetToWord(object):
    def __init__(self, vocabulary):
        vocab = dict((word, index)
                     for index, word in enumerate(vocabulary))
        self.words_by_synset = dict(
            (synset, [vocab[lemma.name] for lemma in synset.lemmas
                      if lemma.name in vocab])
            for synset in wn.all_synsets()
        )

    def all_words_in_relations(self, rels):
        """
        rels: wordnet_rels.Relationships
        """
        words = set()
        for row in rels.data:
            syn_a, syn_b, rel = rels.indices_to_symbolic(row)
            words.update(self.words_by_synset[syn_a])
            words.update(self.words_by_synset[syn_b])
        return words

    def usable(self, syn):
        return bool(self.words_by_synset[syn])

if __name__ == "__main__":
    rels = Relationships()
    import ngrams
    reader = ngrams.NgramReader('/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5', vocab_size=50000)
    s2w = SynsetToWord(reader.word_array)
    indices_in_relationships = s2w.all_words_in_relations(rels)
    print '%d words in vocabulary covered by relationships (out of %d)' % (len(indices_in_relationships) , len(reader.word_array))
