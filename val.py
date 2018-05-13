import json

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt

from loader import load_data
from kb import load_kb

from model import load_tracker_model

CONFIG_FN = 'config.json'

sent_groups = {}

ground_truth = defaultdict(list)
results = defaultdict(list)

def print_ret(utts, utts_seq, states_pred, states_gt):
    for i, utt in enumerate(utts):
        # print utt
        # print utts_seq[i]
        for slot in states_pred:
            _, argmax = states_pred[slot].data[0][i].max(0)
            print '%s pred: %d gt: %d' % (slot, argmax, states_gt[slot].data[i], )

def plot_cfmat(cm):
    plt.figure()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.show()

def evaluate(utts, utts_seq, states_pred, states_gt, sent_groups_pred, onto,  output=False, sent_groups_gt=None):
    count = 0
    correct = 0
    for i, utt in enumerate(utts):
        for slot in states_pred:
            _, argmax = states_pred[slot].data[0][i].max(0)
            count += 1

            results[slot].append(int(argmax))
            ground_truth[slot].append(int(states_gt[slot].data[i]))


            # if slot == 'food':
            #     print "%s %s" % (onto[slot][int(argmax)], onto[slot][int(states_gt[slot].data[i])], )
            if int(argmax) == int(states_gt[slot].data[i]):
                correct += 1
            
        if i == len(utts) - 1:
            break

    return correct, count

def main():
    with open(CONFIG_FN) as f:
        conf = json.load(f)

    global sent_groups

    with open(conf["sent_groups"]) as f:
        sent_groups = json.load(f)["groups"]

    tloader, vloader, testloader, embed, _ = load_data(**conf)

    if not conf["use_pretrained_embedding"]:
        embed = nn.Embedding(embed.num_embeddings, embed.embedding_dim)
        print "use new embedding..."

    kb = load_kb(conf["kb"], 'name')

    model, _ = load_tracker_model(tloader.onto, embed, conf, kb)

    print model

    def test():
        model.eval()
        correct = 0
        count = 0

        states_pred = None

        for utts, _, usr_utts, _, states_gt, kb_found, sys_utt_grp_gt in testloader:
            usr_utts = Variable(usr_utts)
            kb_found = Variable(kb_found)

            states_gt = { slot : Variable(v) for slot, v in states_gt.iteritems() }

            _, _, states_pred, _, sent_grp_pred = model(usr_utts, kb_found, model.state_tracker.init_hidden())
            c, cnt = evaluate(utts, usr_utts, states_pred, states_gt, sent_grp_pred[:-1, :], tloader.onto, output=True)
            correct += c
            count += cnt

        acc = correct / float(count)

        print '-- test correct %d / %d (%.4f) ' % (correct, count, acc, )

        print '-- slot-wise f1-scores'
        for slot in results:
            print '    -- %s %.4f, %.4f %.4f %.4f' % (slot, 
                accuracy_score(ground_truth[slot], results[slot]),
                precision_score(ground_truth[slot], results[slot], average='macro'), 
                recall_score(ground_truth[slot], results[slot], average='macro'),
                f1_score(ground_truth[slot], results[slot], average='macro'))

        conf_mat = confusion_matrix(ground_truth['food'], results['food'])

        print conf_mat
        plot_cfmat(conf_mat)

    test()

if __name__ == '__main__':
    main()
    