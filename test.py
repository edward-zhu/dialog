import json
import sys
import random
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR

from loader import load_data, load_ontology, load_embed, load_sys_vocab, load_kb
from kb import load_kb

from model import load_tracker_model
from decoder import load_generator_model

from codec import Codec

CONFIG_FN = 'config.json'

sent_groups = {}

def print_ret(states_pred, sent_groups_pred):
    for slot in states_pred:
        _, argmax = states_pred[slot].data[0][0].max(0)
        print '%s pred: %d' % (slot, argmax, )

    maxs, argmaxs = sent_groups_pred.data[0][0].topk(3)

    for i, argmax in enumerate(argmaxs):
        print 'sys utt pred: (%d, %.2f)' % (argmax, maxs[i]) + random.choice(sent_groups[str(int(argmax))])

def to_search_criteria(states_pred, onto):
    criteria = []
    for slot in states_pred:
        _, argmax = states_pred[slot].data[0][0].max(0)
        argmax = int(argmax)
        if argmax != 0 and '_req' not in slot:
            criteria.append((slot, onto[slot][argmax - 1]))

    return criteria

def get_kb_result(kb, criteria, indicator_len):
    ret = kb.search_multi(criteria)
    nret = min(len(ret), indicator_len - 1)
    
    vec = torch.zeros(1, indicator_len).long()
    vec[0][nret] = 1
    return ret, Variable(vec)

def main():
    with open(CONFIG_FN) as f:
        conf = json.load(f)

    global sent_groups

    with open(conf["sent_groups"]) as f:
        sent_groups = json.load(f)["groups"]

    kb = load_kb(conf["kb"], 'name')
    sys_vocab, sys_word2idx = load_sys_vocab(conf["sys_vocab"])

    sys_codec = Codec(sys_vocab, sys_word2idx)

    onto, onto_idx = load_ontology(conf["ontology"])

    word2idx, embed = load_embed(**conf)

    usr_codec = Codec([], word2idx)

    trk_model, slot_len_sum = load_tracker_model(onto, embed, conf)

    trk_model.eval()

    hidden = trk_model.state_tracker.init_hidden()
    kb_vec = Variable(torch.zeros(1, conf["kb_indicator_len"]))

    for line in iter(sys.stdin.readline, ''):
        inp = usr_codec.encode(line.strip())

        inp = Variable(torch.LongTensor([ inp, ]))

        sentvecs, states_reps, states_preds, hidden, sent_grp_preds = trk_model(inp, kb_vec, hidden)

        criteria = to_search_criteria(states_preds, onto)
        ret, kb_vec = get_kb_result(kb, criteria, conf["kb_indicator_len"])

        print criteria, kb_vec

        sentvecs = sentvecs.view(1, -1)
        states_reps = states_reps.view(1, -1)

        print_ret(states_preds, sent_grp_preds)

if __name__ == '__main__':
    main()