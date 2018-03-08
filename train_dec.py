
import json
import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR

from loader import load_data
from kb import load_kb

from model import load_tracker_model
from decoder import ConditionNet, Generator, NaiveDecoder

CONFIG_FN = 'config.json'

def main():
    with open(CONFIG_FN) as f:
        conf = json.load(f)

    tloader, vloader, _, embed, sys_vocab = load_data(**conf)

    trk_model, slot_len_sum = load_tracker_model(tloader.onto, embed, conf)

    trk_model.eval()

    sys_vocab_size = len(sys_vocab)

    decoder = NaiveDecoder(sys_vocab)

    cond_net = ConditionNet(conf["sentvec_size"], conf["state_tracker_hidden_size"],
                    slot_len_sum, conf["kb_indicator_len"], conf["cond_size"])
    generator = Generator(conf["cond_size"], conf["generator_hidden_size"], sys_vocab_size, conf["sys_embed_dim"])

    criterion = F.cross_entropy

    cond_net_optim = RMSprop(cond_net.parameters(), conf["lr"], momentum=conf["momentum"])
    generator_optim = RMSprop(generator.parameters(), conf["lr"], momentum=conf["momentum"])

    def save_model(ep):
        model_dir =  conf["model_dir"]
        cond_net_fn = model_dir + "cond_net_%d.model" % (ep)
        gen_fn = model_dir + "gen_%d.model" % (ep)
        with open(cond_net_fn, 'wb') as f:
            torch.save(cond_net.state_dict(), f)
        with open(gen_fn, 'wb') as f:
            torch.save(generator.state_dict(), f)

        print '>>> saved model at %s and %s.' % (cond_net_fn, gen_fn, )

    def train_sent(ep, cond, sys_utt_gt):
        '''train one sentence'''
        hidden = generator.init_hidden()
        inp = Variable(torch.LongTensor([[tloader.sys_word2idx['<sos>']]]))

        teacher_forcing = True if random.random() < conf["teacher_forcing_ratio"] else False

        loss = 0

        sent_out = []

        if teacher_forcing:
            for i in range(sys_utt_gt.size(1) - 1):
                out, hidden = generator(inp, cond, hidden)
                loss += criterion(out.view(1, -1), sys_utt_gt[0][i + 1])
                inp = sys_utt_gt[0][i + 1].view(1, 1)

                topv, topi = out.data.topk(1)
                out_word = int(topi[0][0])

                sent_out.append(out_word)
        else:
            for i in range(sys_utt_gt.size(1) - 1):
                out, hidden = generator(inp, cond, hidden)
                loss += criterion(out.view(1, -1), sys_utt_gt[0][i + 1])

                topv, topi = out.data.topk(1)
                out_word = int(topi[0][0])

                if out_word == tloader.sys_word2idx['<eos>']:
                    break
                inp = Variable(torch.LongTensor([[out_word]]))

                sent_out.append(out_word)

        return loss, sent_out

    def train(ep):
        '''train one dialog'''
        usr_sent, sys_sent, usr_utts, sys_utts, states_gt, kb_found = tloader.next()

        usr_utts = Variable(usr_utts)
        kb_found = Variable(kb_found)

        sentvecs, states_reps, states_preds, _ = trk_model(usr_utts, kb_found, trk_model.state_tracker.init_hidden())

        sentvecs = sentvecs.squeeze()
        states_reps = states_reps.squeeze()

        for slot in states_preds:
            states_preds[slot] = states_preds[slot].squeeze()

        cond = cond_net(sentvecs, states_reps, states_preds, kb_found)

        cond_net_optim.zero_grad()
        generator_optim.zero_grad()

        loss = 0

        out_sents = []

        for i in range(len(sys_utts)):
            # for every sys_utt in this dialog
            sys_utt_gt = Variable(sys_utts[i], requires_grad=False)

            l, o = train_sent(ep, cond[i], sys_utt_gt)
            loss += l
            out_sents.append(decoder.decode(o))

        loss.backward()

        print cond

        if ep % 10 == 0 and ep > 0:
            loss_val = float(loss.data[0]) / float(len(sys_utts))
            print 'epoch %d %.4f' % (ep, loss_val, )
            for i, sys in enumerate(out_sents):
                print '<  %s' % (usr_sent[i], )
                print '>= %s' % (sys_sent[i][6:], )
                print '>? %s' % (sys, )
            print '\n' * 10

        if ep % 1000 == 0 and ep > 0:
            save_model(ep)
            

        for param in cond_net.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in generator.parameters():
            param.grad.data.clamp_(-1, 1)


        cond_net_optim.step()
        generator_optim.step()

    for ep in range(conf["epoch"]):
        train(ep)

if __name__ == '__main__':
    main()


            