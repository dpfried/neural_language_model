from nltk.corpus import wordnet as wn
from relational.wordnet_rels import Relationships

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

if __name__ == "__main__":
    rels = Relationships()
    import ngrams
    reader = ngrams.NgramReader('/cl/nldata/books_google_ngrams_eng/5grams_size3.hd5', vocab_size=50000)
    s2w = SynsetToWord(reader.word_array)
    indices_in_relationships = s2w.all_words_in_relations(rels)
    print '%d words in vocabulary covered by relationships (out of %d)' % (len(indices_in_relationships) , len(reader.word_array))
