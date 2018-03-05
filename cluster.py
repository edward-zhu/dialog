import json

from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def normalize(sent):
    return wordpunct_tokenize(sent.lower())

def cluster_sents(diag_fn):
    with open(diag_fn) as f:
        diags = json.load(f)

    # for grouping sentences
    sentKeys = {}
    sentGroup= []

    stopws = set(stopwords.words('english'))
    for w in ['!',',','.','?','-s','-ly','</s>','s']:
        stopws.add(w)

    # lemmatizer
    lmtzr = WordNetLemmatizer()

    for diag in diags:
        for turn in diag['dialogue']:
            utt = turn['system_transcript']

            words = normalize(utt)

            # sentence group key
            key = tuple(set(sorted(
                [lmtzr.lemmatize(w) for w in words if w not in stopws])))
            if key in sentKeys:
                sentKeys[key][1] += 1
                sentGroup.append( sentKeys[key][0] )
            else:
                sentKeys[key] = [len(sentKeys),1]
                sentGroup.append( sentKeys[key][0] )
        
    # re-assigning sentence group w.r.t their frequency
    mapping = {}
    idx = 0
    cnt = 0
    for key,val in sorted(sentKeys.iteritems(),key=lambda x:x[1][1],reverse=True):
        mapping[val[0]] = idx
        #print idx, val[1], key
        if idx < 70: cnt+=val[1]
        idx += 1

    print sentKeys

if __name__ == '__main__':
    cluster_sents("data/woz2_train.json")
