from nltk.corpus import wordnet as wn
import itertools
import random
from functional import mapcan
import numpy as np

class Relationships(object):
    # names of L{Synset} methods that return other synsets
    relationships = ['hypernyms', 'instance_hypernyms', 'hyponyms', 'instance_hyponyms',
                     'member_holonyms', 'substance_holonyms', 'part_holonyms',
                     'member_meronyms', 'substance_meronyms', 'part_meronyms',
                     'attributes', 'entailments', 'causes', 'also_sees', 'verb_groups',
                     'similar_tos']

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
        a, rel, b = indices_tuple
        return self.synsets[a], self.synsets[b], self.relationships[rel]

    def symbolic_to_indices(self, symbolic_tuple):
        """ convert a tuple of Synset, string, Synset
        into a tuple of indices (i.e. the type stored in self.data)
        """
        a, rel, b = symbolic_tuple
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
