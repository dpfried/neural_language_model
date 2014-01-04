from relational.wordnet_rels import Relationships
from relational.synset_to_word import SynsetToWord
import nltk

if __name__ == "__main__":
    with open('../words.txt') as f:
        vocabulary = set(f.read().split())

    print 'vocab size:', len(vocabulary)
    rels = Relationships()

    s2w = SynsetToWord(vocabulary)
    def synset_contained(synset):
        # s2w, since passed the vocab, will only return words that are in the
        # vocab
        return s2w.words_by_synset[synset] != []

    relationship_counts = nltk.FreqDist()
    used_counts = nltk.FreqDist()

    for tpl in rels.data:
        s1, s2, r = rels.indices_to_symbolic(tpl)
        relationship_counts.inc(r)
        if synset_contained(s1) and synset_contained(s2):
            used_counts.inc(r)

    print 'relationship & WordNet count & in vocab count \\\\'
    for key, rel_count in relationship_counts.iteritems():
        # camel caps and strip final s
        formatted_key = '\\textsc{%s}' % (key.replace('_', ' ').title().replace(' ', ''))[:-1]
        print '%s & %s & %s \\\\' % (formatted_key, rel_count, used_counts[key])
