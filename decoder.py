import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class ConditionNet(nn.Module):
    '''
    Generate condition vector for conditional generation

    cond = [sent_vec, state_rep, slot_states, kb_found]
    '''
    def __init__(self, sent_vec_size, state_tracker_hidden_size, slot_states_len, kb_found_size, cond_size):
        super(ConditionNet, self).__init__()
        self.sent_vec_size = sent_vec_size
        self.state_tracker_hidden_size = state_tracker_hidden_size
        self.slot_states_len = slot_states_len
        self.kb_found_size = kb_found_size
        self.cond_size = cond_size

        self.input_size = self.sent_vec_size + self.state_tracker_hidden_size + self.slot_states_len + self.kb_found_size

        self.fc = nn.Linear(self.input_size, self.cond_size)
        self.bn = nn.BatchNorm1d(self.cond_size)
    
    def forward(self, sent_vec, state_rep, slot_states, kb_found):
        slot_states_rep = torch.cat([ slot_states[slot] for slot in sorted(slot_states.keys())], dim=1)
        slot_states_rep = slot_states_rep.view(state_rep.size(0), -1).float()

        inp = torch.cat([sent_vec, state_rep, slot_states_rep, kb_found.float()], dim=1)
       
        assert(inp.size(1) == self.input_size)

        return F.tanh(self.bn(self.fc(inp)))


class Generator(nn.Module):
    '''
    Generate output sequence distribution given last time input and condition
    '''
    def __init__(self, cond_size, hidden_size, vocab_size, embed_dim):
        super(Generator, self).__init__()

        self.cond_size = cond_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.vocab_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(input_size=self.cond_size + self.vocab_dim, \
                            hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inp, cond, hidden):
        # inp = [[idx, ]]
        # cond = C

        embed = self.embedding(inp) # (1, N words, vocab_dim)

        x = torch.cat([embed, cond.view(1,1,-1)], dim=2).view(1, -1, self.cond_size + self.vocab_dim)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))

class BeamSearchDecoder:
    def __init__(self, vocabs):
        self.vocabs = vocabs

    def decode(out):
        return ""

class NaiveDecoder:
    def __init__(self, vocabs):
        self.vocabs = vocabs

    def decode(self, sent_out):
        '''
        out : N words * vocab_size (p distribution)
        '''
        sent = []

        for w in sent_out:
            sent.append(self.vocabs[w])

        return " ".join(sent)

def load_generator_model(conf, cond_net_fn, gen_fn, slot_len_sum, sys_vocab_size):
    cond_net = ConditionNet(conf["sentvec_size"], conf["state_tracker_hidden_size"],
                    slot_len_sum, conf["kb_indicator_len"], conf["cond_size"])
    generator = Generator(conf["cond_size"], conf["generator_hidden_size"], sys_vocab_size, conf["sys_embed_dim"])

    with open(cond_net_fn, 'rb') as f:
        cond_net.load_state_dict(torch.load(f))

    with open(gen_fn, 'rb') as f:
        generator.load_state_dict(torch.load(f))

    return cond_net, generator

if __name__ == '__main__':
    g = Generator(10, 11, 12, 14)

    print g(Variable(torch.LongTensor([[2]])), Variable(torch.zeros(1, 1, 10)), g.init_hidden())

