import json

import torch
from torch.autograd import Variable
from torch.optim import RMSprop

from loader import load_data
from kb import load_kb

from model import gen_model_and_loss

CONFIG_FN = 'config.json'

def print_ret(utts, utts_seq, states_pred, states_gt):
    for i, utt in enumerate(utts):
        print utt
        # print utts_seq[i]
        for slot in states_pred:
            _, argmax = states_pred[slot].data[0][i].max(0)
            print '%s pred: %d gt: %d' % (slot, argmax, states_gt[slot].data[i], )

def main():
    with open("config.json") as f:
        conf = json.load(f)

    loader, embed = load_data(**conf)

    model, slot_loss = gen_model_and_loss(loader.onto, embed, conf)

    optimizer = RMSprop(model.parameters(), lr=conf["lr"])

    print model

    def train(ep):
        '''train one dialog'''

        model.train()

        optimizer.zero_grad()

        utts, usr_utts, states_gt, kb_found = loader.next()
        
        usr_utts = Variable(usr_utts)
        kb_found = Variable(kb_found)

        states_gt = { slot : Variable(v) for slot, v in states_gt.iteritems() }

        states_reps, states_pred = model(usr_utts, kb_found)

        # print(states_pred)

        loss = slot_loss(states_pred, states_gt)

        if ep % 10 == 0:
            print "epoch %d: %.6f" % (ep, loss / float(len(utts)), )
            print_ret(utts, usr_utts, states_pred, states_gt)

        loss.backward()

        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)

        # torch.nn.utils.clip_grad_norm(model.parameters(), 1)

        optimizer.step()

    for ep in range(conf["epoch"]):
        train(ep)

if __name__ == '__main__':
    main()
    