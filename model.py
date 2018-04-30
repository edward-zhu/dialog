import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import SentenceEncoder
from slot_tracker import InformSlotTracker, RequestSlotTracker
from state_tracker import StateTracker

def gen_tracker_model_and_loss(onto, embed, conf):
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


    model = Model(onto, embed, conf, slot_trackers)
    loss = SlotLoss()

    return model, loss, slot_len_sum

def load_tracker_model(onto, embed, conf):
    model, _, slot_len_sum = gen_tracker_model_and_loss(onto, embed, conf)

    with open(conf["tracker_model"], 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)

    return model, slot_len_sum

class Model(nn.Module):
    '''
    Model

    Encoder word seq -> sentvec_size

    '''
    def __init__(self, onto, embed, conf, slot_trackers):
        super(Model, self).__init__()
        self.sentvec_size = conf["sentvec_size"]
        self.state_tracker_hidden_size = conf["state_tracker_hidden_size"]
        self.sent_group_size = conf["sent_group_size"]

        # sentvec_size = hidden * 2 * output_dim
        self.encoder = SentenceEncoder(embed, conf["vocab_dim"], self.sentvec_size / 2, 1)

        # input = sentvec (sentvec_size) + kb_found (1)
        self.state_tracker = StateTracker(self.sentvec_size + conf["kb_indicator_len"], self.state_tracker_hidden_size)

        self.onto = onto
        
        self.slot_trackers = slot_trackers

        # register slot trackers
        self.slot_trackers_list = nn.ModuleList(self.slot_trackers.values())

        self.sent_group_fc = nn.Linear(self.state_tracker_hidden_size, self.sent_group_size)

    def forward(self, usr_utts, kb_found, state_trk_hidden):
        '''
        N = n_sents
        usr_utts: Variable(torch.LongTensor(N*max_len(sents)))
        '''
        sentvecs = self.encoder(usr_utts)

        # print sentvecs.size()

        # sentvecs: N * sentvec_size
        # kb_found: N * 1

        sentvecs_kb_found = torch.cat((sentvecs, kb_found.view(sentvecs.size(0), -1).float()), dim=1)
        sentvecs_kb_found = sentvecs_kb_found.view(1, sentvecs.size(0), -1)
                                    

        # state_reps: N * state_tracker_hidden_size 
        state_reps, hidden = self.state_tracker(sentvecs_kb_found, state_trk_hidden)

        state_pred = { slot : self.slot_trackers[slot](state_reps) for slot in self.onto }

        sent_group_pred = self.sent_group_fc(state_reps)

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
