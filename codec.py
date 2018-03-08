from common.utils import tokenize
from decoder import NaiveDecoder

class Codec:
    def __init__(self, words, word2idx, decoder=NaiveDecoder, tokenize=tokenize):
        self.words = words
        self.word2idx = word2idx
        self.decoder = decoder(words)
        self.tokenize = tokenize

    def encode(self, sent):
        tokens = self.tokenize(sent.lower())
        return [ self.word2idx.get(t, 0) for t in tokens ]

    def decode(self, seq):
        return self.decoder.decode(seq)