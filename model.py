import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import SentenceEncoder
from slot_tracker import InformSlotTracker, RequestSlotTracker
from state_tracker import StateTracker

from torch.autograd import Variable

def gen_tracker_model_and_loss(onto, embed, conf, kb):
    state_tracker_hidden_size = conf["state_tracker_hidden_size"]

    slot_trackers = {}

    slot_len_sum = 0

    for slot in onto:
        if len(onto[slot]) > 2:
            slot_trackers[slot] = \
                InformSlotTracker(state_tracker_hidden_size, len(onto[slot]))
            slot_len_sum += len(onto[slot]) + 1
        else:
            slot_trackers[slot] = \
                RequestSlotTracker(state_tracker_hidden_size)
            slot_len_sum += 2 


    model = Model(onto, embed, conf, slot_trackers, kb)
    loss = SlotLoss()

    return model, loss, slot_len_sum

def load_tracker_model(onto, embed, conf, kb):
    model, _, slot_len_sum = gen_tracker_model_and_loss(onto, embed, conf, kb)

    with open(conf["tracker_model"], 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)

    return model, slot_len_sum

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

class Model(nn.Module):
    '''
    Model

    Encoder word seq -> sentvec_size

    '''
    def __init__(self, onto, embed, conf, slot_trackers, kb):
        super(Model, self).__init__()
        self.sentvec_size = conf["sentvec_size"]
        self.state_tracker_hidden_size = conf["state_tracker_hidden_size"]
        self.sent_group_size = conf["sent_group_size"]

        # sentvec_size = hidden * 2 * output_dim
        self.encoder = SentenceEncoder(embed, conf["vocab_dim"], self.sentvec_size / 2, 1)

        # input = sentvec (sentvec_size) + kb_found (1)
        self.state_tracker = StateTracker(self.sentvec_size, self.state_tracker_hidden_size)

        self.onto = onto
        
        self.slot_trackers = slot_trackers

        # register slot trackers
        self.slot_trackers_list = nn.ModuleList(self.slot_trackers.values())

        self.sent_group_fc = nn.Linear(self.state_tracker_hidden_size + conf["kb_indicator_len"], self.sent_group_size)


        self.kb_indicator_len = conf["kb_indicator_len"]

        self.kb = kb

    def forward(self, usr_utts, kb_found, state_trk_hidden):
        '''
        N = n_sents
        usr_utts: Variable(torch.LongTensor(N*max_len(sents)))
        '''
        sentvecs = self.encoder(usr_utts)

        # print sentvecs.size()

        # sentvecs: N * sentvec_size
        # kb_found: N * 1

        sentvecs = sentvecs.unsqueeze(0)

        # state_reps: N * state_tracker_hidden_size 
        state_reps, hidden = self.state_tracker(sentvecs, state_trk_hidden)

        state_pred = { slot : self.slot_trackers[slot](state_reps) for slot in self.onto }

        if kb_found is None:
            criteria = to_search_criteria(state_pred, self.onto)
            ret, kb_found = get_kb_result(self.kb, criteria, self.kb_indicator_len)

        statereps_kb_found = torch.cat((state_reps.squeeze(0), kb_found.float()), dim=1)

        sent_group_pred = self.sent_group_fc(statereps_kb_found)

        return sentvecs, state_reps, state_pred, hidden, sent_group_pred

class SlotLoss(nn.Module):
    def __init__(self):
        super(SlotLoss, self).__init__()

    def forward(self, pred, gt):
        loss = 0
        for k, v in pred.iteritems():
            v = v.squeeze()
            loss += F.cross_entropy(v, gt[k])
        
        return loss
