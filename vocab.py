from collections import defaultdict

UNKNOWN_WORD = '*UNKNOWN*'
class Vocabulary(object):
    """lexicon with an unknown word represented at index 0"""
    def __init__(self, words):
        self.words = words
        self.N = len(words)
        self.symbol_to_index = defaultdict(int,
                                           dict((word, index) for index, word in enumerate(words)))
        self.index_to_symbol = defaultdict(lambda : UNKNOWN_WORD,
                                           dict((index, word) for index, word in enumerate(words)))

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.index_to_symbol[index]

    def __iter__(self):
        return self.words.__iter__()
