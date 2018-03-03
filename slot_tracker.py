import torch
from torch import nn

class InformSlotTracker(nn.Module):
    '''
        Informable Slot Tracker

        get slot value distribution from state at time t

        e.g. price=cheap

        input: state tracker output `state_t`
        output: value distribution `P(v_s_t| state_t)`
    '''
    def __init__(self, input_dim, n_choices):
        super(InformSlotTracker, self).__init__()
        self.n_choices = n_choices + 1 # include don't care
        self.fc = nn.Linear(input_dim, self.n_choices)

    def forward(self, state):
        return self.fc(state)

class RequestSlotTracker(nn.Module):
    '''
        Requestable Slot Tracker

        get a request type activation state distribution from state at time t

        e.g.
            address=1 (currently address is requested)
            phone=0 (currently phone is not cared by the user)

        input: state tracker output `state_t`
        output: value binary distribution `P(v_s_t| state_t)`
    '''
    def __init__(self, input_dim):
        super(RequestSlotTracker, self).__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, state):
        return self.fc(state)
