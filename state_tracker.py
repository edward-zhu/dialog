import torch
import torch.nn as nn
import torch.functional as F

from torch.autograd import Variable

class StateTracker(nn.Module):

    def __init__(self, feat_size, hidden_size):
        super(StateTracker, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=feat_size,
                            hidden_size=hidden_size, batch_first=True, dropout=0.9)

    def forward(self, inp, hidden):
        out, h_n = self.gru(inp, hidden)

        return out, h_n

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))