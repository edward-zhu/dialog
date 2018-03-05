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

def evaluate(utts, utts_seq, states_pred, states_gt):
    count = 0
    correct = 0
    for i, utt in enumerate(utts):
        for slot in states_pred:
            _, argmax = states_pred[slot].data[0][i].max(0)
            count += 1

            if int(argmax) == int(states_gt[slot].data[i]):
                correct += 1
    return correct, count

def main():
    with open("config.json") as f:
        conf = json.load(f)

    tloader, vloader, _, embed = load_data(**conf)

    model, slot_loss = gen_model_and_loss(tloader.onto, embed, conf)

    optimizer = RMSprop(model.parameters(), lr=conf["lr"])

    print model

    global best_acc, best_ep, no_improve

    best_acc = 0.0
    best_ep = 0
    no_improve = 0

    def train(ep):
        '''train one dialog'''

        model.train()

        optimizer.zero_grad()

        utts, usr_utts, states_gt, kb_found = tloader.next()
        
        usr_utts = Variable(usr_utts)
        kb_found = Variable(kb_found)

        states_gt = { slot : Variable(v) for slot, v in states_gt.iteritems() }

        states_reps, states_pred = model(usr_utts, kb_found)

        # print(states_pred)

        loss = slot_loss(states_pred, states_gt)

        if ep % 10 == 0:
            c, cnt = evaluate(utts, usr_utts, states_pred, states_gt)
            print "    epoch %d: %.6f correct %d / %d (%.4f)" % (ep, loss / float(len(utts)), c, cnt, c / float(cnt))
            # print_ret(utts, usr_utts, states_pred, states_gt)

        loss.backward()

        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)

        # torch.nn.utils.clip_grad_norm(model.parameters(), 1)

        optimizer.step()

    def validate(ep):
        model.eval()
        vloader.reset()

        global best_acc, best_ep, no_improve

        correct = 0
        count = 0

        for utts, usr_utts, states_gt, kb_found in vloader:
            usr_utts = Variable(usr_utts)
            kb_found = Variable(kb_found)

            states_gt = { slot : Variable(v) for slot, v in states_gt.iteritems() }

            states_reps, states_pred = model(usr_utts, kb_found)
            c, cnt = evaluate(utts, usr_utts, states_pred, states_gt)
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

        print 'epoch %d: val correct %d / %d (%.4f) best %.4f' % (ep, correct, count, acc, best_acc, )
        

    for ep in range(conf["epoch"]):
        train(ep)
        if ep % 100 == 1:
            validate(ep)

        if no_improve > 10:
            print "-> no improve > 10 times exit."
            exit()

if __name__ == '__main__':
    main()
    