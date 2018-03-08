import json
import sys
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

MAX_SENT_LEN = 20
CONFIG_FN = 'config.json'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cond-net", type=str, required=True)
    parser.add_argument("--gen-net", type=str, required=True)
    return parser.parse_args()

def print_ret(states_pred):
    for slot in states_pred:
        _, argmax = states_pred[slot].data[0][0].max(0)
        print '%s pred: %d' % (slot, argmax, )

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

    args = parse_args()

    kb = load_kb(conf["kb"], 'name')
    sys_vocab, sys_word2idx = load_sys_vocab(conf["sys_vocab"])

    sys_codec = Codec(sys_vocab, sys_word2idx)

    onto, onto_idx = load_ontology(conf["ontology"])

    word2idx, embed = load_embed(**conf)

    usr_codec = Codec([], word2idx)

    trk_model, slot_len_sum = load_tracker_model(onto, embed, conf)
    cond_net, generator = load_generator_model(conf, args.cond_net, args.gen_net, slot_len_sum, len(sys_vocab))

    cond_net.eval()
    generator.eval()

    trk_model.eval()

    hidden = trk_model.state_tracker.init_hidden()
    kb_found = Variable(torch.zeros(1, conf["kb_indicator_len"]))

    def gen_sent(cond):
        '''train one sentence'''
        hidden = generator.init_hidden()
        inp = Variable(torch.LongTensor([[sys_word2idx['<sos>']]]))

        sent_out = []

        for i in range(MAX_SENT_LEN):
            out, hidden = generator(inp, cond, hidden)

            topv, topi = out.data.topk(1)
            out_word = int(topi[0][0])

            if out_word == sys_word2idx['<eos>']:
                break
            inp = Variable(torch.LongTensor([[out_word]]))

            sent_out.append(out_word)

        return sys_codec.decode(sent_out)

    for line in iter(sys.stdin.readline, ''):
        inp = usr_codec.encode(line.strip())

        inp = Variable(torch.LongTensor([ inp, ]))

        sentvecs, states_reps, states_preds, hidden = trk_model(inp, kb_found, hidden)

        # print_ret(states_preds)
        criteria = to_search_criteria(states_preds, onto)
        ret, kb_vec = get_kb_result(kb, criteria, conf["kb_indicator_len"])

        sentvecs = sentvecs.view(1, -1)
        states_reps = states_reps.view(1, -1)

        for slot in states_preds:
            states_preds[slot] = states_preds[slot].view(1, -1)
        
        cond = cond_net(sentvecs, states_reps, states_preds, kb_found)

        print gen_sent(cond)

if __name__ == '__main__':
    main()