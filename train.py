import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR

from loader import load_data
from kb import load_kb

from model import gen_tracker_model_and_loss

CONFIG_FN = 'config.json'

sent_groups = {}

def print_ret(utts, utts_seq, states_pred, states_gt):
    for i, utt in enumerate(utts):
        # print utt
        # print utts_seq[i]
        for slot in states_pred:
            _, argmax = states_pred[slot].data[0][i].max(0)
            print '%s pred: %d gt: %d' % (slot, argmax, states_gt[slot].data[i], )

def evaluate(utts, utts_seq, states_pred, states_gt, sent_groups_pred, sent_groups_gt=None):
    count = 0
    correct = 0
    for i, utt in enumerate(utts):
        for slot in states_pred:
            _, argmax = states_pred[slot].data[0][i].max(0)
            count += 1

            if int(argmax) == int(states_gt[slot].data[i]):
                correct += 1
            
        if i == len(utts) - 1:
            break

        print utt
        _, argmax = sent_groups_pred.data[i].max(0)
        print 'sys utt pred: (%d)' % (int(argmax)) + sent_groups[str(int(argmax))][0]
        if sent_groups_gt is not None:
            print 'sys utt gt  (%d): ' % (int(sent_groups_gt[i])) + sent_groups[str(int(sent_groups_gt[i]))][0]

    return correct, count

def main():
    with open(CONFIG_FN) as f:
        conf = json.load(f)

    global sent_groups

    with open(conf["sent_groups"]) as f:
        sent_groups = json.load(f)["groups"]

    tloader, vloader, _, embed, _ = load_data(**conf)

    if not conf["use_pretrained_embedding"]:
        embed = nn.Embedding(embed.num_embeddings, embed.embedding_dim)
        print "use new embedding..."

    kb = load_kb(conf["kb"], 'name')

    model, slot_loss, _ = gen_tracker_model_and_loss(tloader.onto, embed, conf, kb)

    optimizer = RMSprop(model.parameters(), lr=conf["lr"])
    scheduler = StepLR(optimizer, step_size=100, gamma=0.99)

    print model

    # exit()

    global best_acc, best_ep, no_improve

    best_acc = 0.0
    best_ep = 0
    no_improve = 0

    def train(ep):
        '''train one dialog'''
        model.train()

        optimizer.zero_grad()

        # ignore sys utts
        # user utts (string), _, user utts (idx sequence), _, states in this turn, kb_found indicator 
        utts, sys, usr_utts, _, states_gt, kb_found, sys_utt_grp_gt = tloader.next()
        
        usr_utts = Variable(usr_utts)
        kb_found = Variable(kb_found)
        sys_utt_grp_gt = Variable(sys_utt_grp_gt)

        states_gt = { slot : Variable(v) for slot, v in states_gt.iteritems() }

        _, states_reps, states_pred, _, sent_grp_pred = model(usr_utts, kb_found, model.state_tracker.init_hidden())

        # print(states_pred)

        loss = slot_loss(states_pred, states_gt)
        loss += F.cross_entropy(sent_grp_pred[:-1, :].view(-1, 200), sys_utt_grp_gt) * 0.2

        if ep % 100 == 0:
            c, cnt = evaluate(utts, usr_utts, states_pred, states_gt, sent_grp_pred[:-1, :], sys_utt_grp_gt)
            print "    epoch %d: %.6f correct %d / %d (%.4f)" % (ep, loss / float(len(utts)), c, cnt, c / float(cnt))

        loss.backward()

        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)

        optimizer.step()

    def validate(ep):
        model.eval()
        vloader.reset()

        global best_acc, best_ep, no_improve

        correct = 0
        count = 0

        for utts, _, usr_utts, _, states_gt, kb_found, sys_utt_grp_gt in vloader:
            usr_utts = Variable(usr_utts)
            kb_found = Variable(kb_found)

            states_gt = { slot : Variable(v) for slot, v in states_gt.iteritems() }

            _, states_reps, states_pred, _, sent_grp_pred = model(usr_utts, kb_found, model.state_tracker.init_hidden())
            c, cnt = evaluate(utts, usr_utts, states_pred, states_gt, sent_grp_pred[:-1, :])
            correct += c
            count += cnt

        acc = correct / float(count)
        if acc > best_acc:
            best_acc = acc
            best_ep = ep
            no_improve = 0

            model_path = conf["model_dir"] + "tracker_%d.model" % (ep, )
            with open(model_path, 'wb') as f:
                print '-> saved best model at %s' % (model_path, )
                torch.save(model.state_dict(), f)
        else:
            no_improve += 1

        print 'epoch %d: val correct %d / %d (%.4f) best %.4f (epoch %d)' % (ep, correct, count, acc, best_acc, best_ep, )
        

    for ep in range(conf["epoch"]):
        scheduler.step()
        train(ep)
        if ep % 100 == 1:
            validate(ep)

        if no_improve > conf["early_stopping"]:
            print "-> no improve > 10 times exit."
            exit()

if __name__ == '__main__':
    main()
    