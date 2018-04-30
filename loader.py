import sys
import json
import pprint
import random
import pickle

from collections import defaultdict

import torch
import torch.nn as nn

import numpy as np

from kb import load_kb
from common.utils import pad_sequence, tokenize

Debug = 0

if Debug > 0:
    from torch.autograd import Variable

def load_dialogs(diag_fn, kb, groups_fn=None):
    with open(diag_fn) as f:
        diags = json.load(f)

    group_iter = None
    if groups_fn is not None:
        with open(groups_fn) as f:
            groups = json.load(f)["labels"] 
        group_iter = iter(groups)

    data = []

    for diag in diags:
        usr_utts = []
        sys_utts = []
        states = []
        kb_found = []

        sys_utt_grps = []

        for turn in diag['dialogue']:
            usr_utts.append(turn['transcript'])
            sys_utts.append("<sos> " + turn['system_transcript'] + " <eos>")

            slots = []
            search_keys = []

            for state in turn['belief_state']:
                if state["act"] == "inform":
                    slots.append(state["slots"][0])
                    state["slots"][0][0] = state["slots"][0][0].replace(" ", "").replace("center", "centre")
                    search_keys.append(state["slots"][0])
                elif state["act"] == 'request':
                    slots.append((state["slots"][0][1].replace(" ", "") + "_req", "care")) 
                else:
                    raise RuntimeError("illegal state : %s" % (state, ))
            
            states.append(slots)

            ret = kb.search_multi(search_keys)
            kb_found.append(len(ret))

        sys_utts = sys_utts[1:] # the first sys_utt is always empty

        if group_iter is not None:
            for _ in sys_utts:
                group = next(group_iter)
                sys_utt_grps.append(group)

        data.append({
            'usr_utts': usr_utts,
            'sys_utts':sys_utts,
            'sys_utt_grps' : sys_utt_grps,
            'states':states,
            'kb_found':kb_found,
        })

    if Debug > 0:
        pprint.pprint(data[0])


    return data

def load_sys_vocab(sys_vocab_fn):
    with open(sys_vocab_fn) as f:
        vocabs = json.load(f)

    vocabs = vocabs["vocabs"]
    
    return vocabs, { w : i for i, w in enumerate(vocabs) }


'''
Load vocabulary

ref: https://github.com/A-Jacobson/CNN_Sentence_Classification/blob/master/WordVectors.ipynb

if out_fn is set to a filename, it can save generated embedding model to it.
'''
def load_vocab_and_embedding(vocab_fn, dim=300, strip=0, max_len=20, subset=None, out_fn=None):
    vocab = {"<null>" : np.zeros(dim), }
    word2idx = {'<null>' : 0, }

    idx = 1

    import re

    # a naive filter for those apparently non-sense words
    nopattern = re.compile(r".*[.:/,_\\|!?\d].*|.*[-][-]+.*")

    with open(vocab_fn) as f:
        for line in f.readlines():
            values = line.split()
            word = values[0]

            if strip > 0:
                word = word[strip:]

            if len(word) > max_len:
                continue

            if len(word) > 1 and nopattern.match(word):
                continue

            if subset is not None and (word not in subset):
                continue
            
            vector = np.array(values[1:], dtype='float32')
            vocab[word] = vector
            word2idx[word] = idx
            idx += 1
            
            if Debug > 0:
                sys.stdout.write('add %d word -> %s\r' % (idx, word,))

    print '-'

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
        
        # info slot can also be req slot
        k = k + "_req"
        onto[k] = values + ['dontcare']
        onto_idx[k] = {
            'dontcare' : 0,
            'care' : 1,
        }

    req_data = data["requestable"]
    
    for k, values in req_data.iteritems():
        k = k + "_req"
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

    def __init__(self, diags, word2idx, sys_word2idx, onto, onto_idx, kb_fonud_len=5, mode='train'):
        self.diags = diags

        self.word2idx = word2idx
        self.sys_word2idx = sys_word2idx
        
        self.cur = 0

        self.onto = onto
        self.onto_idx = onto_idx

        self.kb_found_len = kb_fonud_len
        self.kb_indicator = torch.eye(kb_fonud_len + 2).long()

        self.mode = mode

    def get_vocabs(self):
        vocabs = []
        sys_vocabs = []
        for diag in self.diags:
            for s in diag['usr_utts']:
                # print s, self._gen_utt_seq(s)
                vocabs.extend(self._sent_normalize(s))
            for s in diag['sys_utts']:
                print self._sent_normalize(s)
                sys_vocabs.extend(self._sent_normalize(s))
            
        return set(vocabs), set(sys_vocabs)
        
    def _get(self, i):
        diag = self.diags[i]
        usr_utts = [ self._gen_utt_seq(self.word2idx, s) for s in diag['usr_utts']]
        usr_utts = torch.LongTensor(pad_sequence(usr_utts))
        states = self._gen_state_vecs(diag['states'])
        kb_found = torch.cat([ self.kb_indicator[x].view(1, -1) 
                            if x <= self.kb_found_len else self.kb_indicator[self.kb_found_len + 1].view(1, -1) 
                            for x in diag['kb_found']])
        sys_utts = [ self._gen_utt_seq(self.sys_word2idx, s)for s in diag['sys_utts']]
        sys_utts = [ torch.LongTensor(utt).view(1, -1) for utt in sys_utts] # B (=1) * N words

        sys_utt_grps = torch.LongTensor(diag['sys_utt_grps'])

        return diag['usr_utts'], diag['sys_utts'], usr_utts, sys_utts, states, kb_found, sys_utt_grps

    def _sent_normalize(self, sent):
        return tokenize(sent.lower())

    def _gen_utt_seq(self, word2idx, utt):
        '''
        convert string to word idx seq
        input: word2idx -> { word: # }
        '''
        utt = self._sent_normalize(utt)

        # unknown word is assigned <nuk>
        utt = [ word2idx.get(x, 0) for x in utt]

        return utt

    def _gen_state_vecs(self, states):
        '''
        from: [,
            [
                ['info_slot', 'value'],
                ['req_slot', 'care/dontcare']
            ],
            ...
        ]

        to: {
            'info_slot': LongTensor([4, 0, 3, 5]),
            'req_slot': LongTensor([0, 1, 0, 1])
        }
        '''

        state_vecs = { slot: torch.zeros(len(states)).long() for slot in self.onto }
        for t, states_at_time_t in enumerate(states):
            for s, v in states_at_time_t:
                state_vecs[s][t] = self.onto_idx[s][v]
        
        return state_vecs

    def __iter__(self):
        return self

    def reset(self):
        self.cur = 0

    def next(self):
        '''
        get one dialog training data

        per turn:
        - user utts ([sentence, ])
        - sys utts ([sentence, ]])
        - user utt (LongTensor(N sentences * N words (in word idx)))
        - sys utt ([ LongTensor(B (=1) * N words (in word idx)) , ... N sentences])
        - states
            - informable: value #
            - requestable: 0 / 1
        - kb_found
        '''
        ret = self._get(self.cur)

        self.cur += 1
        if self.cur == len(self.diags):
            if self.mode == 'test':
                raise StopIteration()
            random.shuffle(self.diags)
            self.cur = 0

        return ret
        
def load_embed_model(fn, embed_dim=300):
    with open(fn + '.w2i', 'rb') as f:
        word2idx = pickle.load(f)

    embed = nn.Embedding(len(word2idx), embed_dim)
    embed.load_state_dict(torch.load(fn))
    return word2idx, embed

def load_embed(**kargs):
    if not kargs["generate_embed"]:
        word2idx, embed = load_embed_model(kargs["embed_model"])
    else:
        word2idx, embed = load_vocab_and_embedding(kargs["paragram_file"], out_fn=kargs["embed_model"])

    return word2idx, embed


def load_data(**kargs):
    kb = load_kb(kargs["kb"], "name")

    word2idx, embed = load_embed(**kargs)

    sys_vocab, sys_word2idx = load_sys_vocab(kargs["sys_vocab"])
    
    diags_train = load_dialogs(kargs["diags_train"], kb, kargs["sent_groups"])
    diags_val = load_dialogs(kargs["diags_val"], kb)
    diags_test = load_dialogs(kargs["diags_test"], kb)

    onto, onto_idx = load_ontology(kargs["ontology"])

    return DataLoader(diags_train, word2idx, sys_word2idx, onto, onto_idx), \
            DataLoader(diags_val, word2idx, sys_word2idx, onto, onto_idx, mode='test'), \
            DataLoader(diags_test, word2idx, sys_word2idx, onto, onto_idx, mode='test'), embed, sys_vocab


if __name__ == '__main__':
    with open("config.json") as f:
        conf = json.load(f)

    '''   
    tloader, vloader, testloader, embed, sys_vocab = load_data(**conf)

    print tloader.next()
    '''

    '''
    union_subset = set(vloader.get_vocabs()[1]).union(set(tloader.get_vocabs()[1]))

    with open("sys_subset.json", "w") as f:
        json.dump({
            "vocabs" : list(union_subset)
        }, f)

    # pprint.pprint(loader.next())
    '''

    '''
    with open("subset.json") as f:
        subset = set(json.load(f)['vocabs'])
        load_vocab_and_embedding(conf["paragram_file"], subset=subset, out_fn="data/vocab_tiny.model")
    '''