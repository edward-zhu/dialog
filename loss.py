import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotLoss(nn.Module):
    def __init__(self, slot_trackers):
        super(SlotLoss, self).__init__()

        self.slot_trackers = slot_trackers

    def forward(self, out, y):
        for trk in self.slot_trackers:
            pass