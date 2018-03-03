import json
import pprint
import random
import pickle

from nltk.tokenize import RegexpTokenizer

from collections import defaultdict

import torch
import torch.nn as nn

import numpy as np

from kb import load_kb

Debug = 1

if Debug > 0:
    from torch.autograd import Variable

def load_dialogs(diag_fn, kb):
    with open(diag_fn) as f:
        diags = json.load(f)

    data = []

    for diag in diags:
        usr_utts = []
        sys_utts = []
        states = []
        kb_found = []

        for turn in diag['dialogue']:
            usr_utts.append(turn['transcript'])
            sys_utts.append(turn['system_transcript'])

            slots = []
            search_keys = []

            for state in turn['belief_state']:
                if state["act"] == "inform":
                    slots.append(state["slots"][0])
                    state["slots"][0][0] = state["slots"][0][0].replace(" ", "")
                    search_keys.append(state["slots"][0])
                elif state["act"] == 'request':
                    slots.append((state["slots"][0][1], "care"))
                else:
                    raise RuntimeError("illegal state : %s" % (state, ))
            
            states.append(slots)

            ret = kb.search_multi(search_keys)
            kb_found.append(len(ret))

        sys_utts = sys_utts[1:] # the first sys_utt is always empty

        data.append({
            'usr_utts': usr_utts,
            'sys_utts':sys_utts,
            'states':states,
            'kb_found':kb_found,
        })

    if Debug > 0:
        pprint.pprint(data[0])


    return data

'''
Load vocabulary

ref: https://github.com/A-Jacobson/CNN_Sentence_Classification/blob/master/WordVectors.ipynb

if out_fn is set to a filename, it can save generated embedding model to it.
'''
def load_vocab_and_embedding(vocab_fn, dim=300, strip=True, out_fn=None):
    vocab = {"<null>" : np.zeros(dim), }
    word2idx = {'<null>' : 0, }

    with open(vocab_fn) as f:
        for idx, line in enumerate(f.readlines()):
            values = line.split()
            word = values[0]
            if strip:
                word = word[3:]
            vector = np.array(values[1:], dtype='float32')
            vocab[word] = vector
            word2idx[word] = idx + 1

    embeddings = np.zeros((len(vocab), dim))
    for word in vocab:
        embeddings[word2idx[word]] = vocab[word]

    if Debug:
        test_seq = [word2idx[x] for x in ["i", "am"]]
        vec = np.array([vocab[x] for x in ["i", "am"]])
        vec = torch.from_numpy(vec)

    embeddings = torch.from_numpy(embeddings).float()

    embed = nn.Embedding(len(vocab), dim)
    embed.weight = nn.Parameter(embeddings)

    if Debug:
        vec2 = embed(Variable(torch.LongTensor(test_seq)))
        assert(torch.sum(vec - vec2.data) < 0.001)

    if out_fn is not None:
        torch.save(embed.state_dict(), out_fn)
        with open(out_fn + '.w2i', 'wb') as f:
            pickle.dump(word2idx, f)

        if Debug:
            print 'saved embedding model to: ' + out_fn
    
    return word2idx, embed

def load_ontology(fn):
    with open(fn) as f:
        data = json.load(f)

    onto = {}
    onto_idx = defaultdict(dict)

    # informable slots
    inf_data = data["informable"]

    for k, values in inf_data.iteritems():
        onto[k] = values + ['dontcare']
        onto_idx[k]['dontcare'] = 0
        for v in values:
            onto_idx[k][v] = len(onto_idx[k])

    req_data = data["requestable"]
    
    for k, values in req_data.iteritems():
        onto[k] = values + ['dontcare']
        onto_idx[k] = {
            'dontcare' : 0,
            'care' : 1,
        }

    return onto, onto_idx

class DataLoader:
    '''
    DataLoader

    a dialog data warehose
    '''

    def __init__(self, diags, word2idx, onto, onto_idx):
        self.diags = diags
        self.word2idx = word2idx
        self.cur = 0

        self.onto = onto
        self.onto_idx = onto_idx

        self.tokenizer = RegexpTokenizer(r'\w+')
        
    def _get(self, i):
        diag = self.diags[i]
        usr_utts = diag['usr_utts']
        states = [ self._gen_state_for_input(s) for s in diag['states']]
        kb_found = [ 0 if x == 0 else 1 for x in diag['kb_found']]

        return diag, usr_utts, states, kb_found

    def _sent_normalize(self, sent):
        return self.tokenizer.tokenize(sent.lower())

    def _gen_utt_seq(self, utt):
        '''convert string to word idx seq'''
        utt = self._sent_normalize(utt)
        utt = [ self.word2idx.get(x, 0) for x in utt]
        return utt

    def _gen_state_for_input(self, state_list):
        '''
        from: [
            ['info_slot', 'value'],
            ['req_slot', 'care/dontcare']
        ]

        to: {
            'info_slot': 1/2/3...,
            'req_slot': 1/0
        }
        '''
        states = {k : 0 for k in self.onto}
        for s, v in state_list:
            states[s] = self.onto_idx[s][v]

        return states

    def next(self):
        '''
        get one dialog training data

        per turn:
        - user utt (in array of word indexes)
        - states
            - informable: value #
            - requestable: 0 / 1
        - kb_found
        '''
        ret = self._get(self.cur)

        self.cur += 1
        if self.cur == len(self.diags):
            random.shuffle(self.diags)

        return ret
        
def load_embed_model(fn, embed_dim=300):
    with open(fn + '.w2i', 'rb') as f:
        word2idx = pickle.load(f)

    embed = nn.Embedding(len(word2idx), embed_dim)
    embed.load_state_dict(torch.load(fn))
    return word2idx, embed

def load_data(**kargs):
    kb = load_kb(kargs["kb"], "name")

    if "embed_model" in kargs:
        word2idx, embed = load_embed_model(kargs["embed_model"])
    else:
        word2idx, embed = load_vocab_and_embedding(kargs["paragram_file"])
    
    diags = load_dialogs(kargs["diags"], kb)

    onto, onto_idx = load_ontology(kargs["ontology"])

    return DataLoader(diags, word2idx, onto, onto_idx), embed


if __name__ == '__main__':
    with open("config.json") as f:
        conf = json.load(f)

    loader, embed = load_data(**conf)

    pprint.pprint(loader.next())