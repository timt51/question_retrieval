from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # x_positive, y: (batch_sample_index, encoding_len)
        # x_negatives: (negative_query_index, batch_sample_index, encoding_len)
        num_negative_queries = x_negatives.size(0)
        positive_similarity = F.cosine_similarity(x_positive, y, dim=1)
        negative_similarities = F.cosine_similarity(
            x_negatives, y.repeat(num_negative_queries, 1, 1), dim=2)
        # positive_similarity: (batch_sample_index)
        # negative_similarities: (negative_query_index, batch_sample_index)
        negative_similarity, _ = torch.max(negative_similarities, dim=0)
        negative_similarity = torch.add(negative_similarity, self.margin)
        return torch.max(negative_similarity.sub(positive_similarity))

def mean_average_precision():
    pass

def mean_reciprocal_rank():
    pass

def precision_at_n(n):
    pass
