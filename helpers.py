from collections import namedtuple

import torch.nn as nn

Result = namedtuple("Result", \
        "model lr other_hyperparameters map mrr pat1 pat5")

class MaxMarginLoss(nn.Module):
    def __init__(self, margin):
        super(MaxMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, x_positive, x_negatives, y):
        """
        Computes max margin loss.

        Keyword arguments:
        x_positive -- encoding of the positive sample
        x_negative -- encodings of the negative samples
        y -- encoding of query
        """
        pass

def mean_average_precision():
    pass

def mean_reciprocal_rank():
    pass

def precision_at_n(n):
    pass
