import torch.nn as nn

class MaxMarginLoss(nn.Module):
    def __init__(self, margin):
        super(MaxMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, x, y):
        pass

def mean_average_precision():
    pass

def mean_reciprocal_rank():
    pass

def precision_at_n(n):
    pass
