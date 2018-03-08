from nltk import wordpunct_tokenize
from nltk.tokenize import RegexpTokenizer

def pad_sequence(seqs):
    max_len = max([len(seq) for seq in seqs])

    padded = [seq + [0] * (max_len - len(seq)) for seq in seqs]

    return padded

tokenizer = RegexpTokenizer(r'<[a-z][.\w]+>|[^<]+')

def tokenize(sent):
    tokens = tokenizer.tokenize(sent)
    ret = []
    for t in tokens:
        if '<' not in t:
            ret.extend(wordpunct_tokenize(t))
        else:
            ret.append(t)
    return ret

if __name__ == '__main__':
    print tokenize("it's <v.pricerange> ly priced. is there anything else i can do for you?")



