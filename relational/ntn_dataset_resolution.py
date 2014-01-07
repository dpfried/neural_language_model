from relational.wordnet_rels import relationship_methods
from nltk.corpus import wordnet as wn
from collections import defaultdict

import nltk

default_filename = '/home/dfried/naist/Wordnet/train.txt'

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
