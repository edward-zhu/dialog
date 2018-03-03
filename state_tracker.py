import torch
import torch.nn as nn
import torch.functional as F

class StateTracker(nn.Module):

    def __init__(self, feat_size, hidden_size):
        super(StateTracker, self).__init__()
        self.gru = nn.GRU(input_size=feat_size, hidden_size=hidden_size)

    def forward(self, inp):
        out, h_n = self.gru(inp)

        return h_n