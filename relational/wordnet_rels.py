from nltk.corpus import wordnet as wn
import random
from functional import mapcan
import numpy as np

from relational.ntn_dataset_resolution import default_filename, resolve_file
from relational.synset_to_word import SynsetToWord

# names of L{Synset} methods that return other synsets
relationship_methods = ['hypernyms', 'instance_hypernyms', 'hyponyms', 'instance_hyponyms',
                        'member_holonyms', 'substance_holonyms', 'part_holonyms',
                        'member_meronyms', 'substance_meronyms', 'part_meronyms',
                        'attributes', 'entailments', 'causes', 'also_sees', 'verb_groups',
                        'similar_tos']

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

if __name__ == "__main__":
    rel = Relationships()
