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
        negative_similarity = torch.clamp(negative_similarity.sub(positive_similarity), min=0)
        return torch.mean(negative_similarity)

def recall(relevant, retrieved):
    """
    Computes the recall value for the set of retrieved elements
    against the set of relevant documents

    :param relevant: a set of desired documents
    :param retrieved: a set of retrieved documents
    :return: the precision rate, a real number from [0,1]
    """
    num_in_common = float(len(relevant & retrieved))
    return num_in_common / len(relevant)


def precision(relevant, retrieved):
    """
    Computes the precision value for the set of retrieved elements
    against the set of relevant documents

    :param relevant: a set of desired documents
    :param retrieved: a set of retrieved documents
    :return: the precision rate, a real number from [0,1]
    """
    num_in_common = float(len(relevant & retrieved))
    return num_in_common / len(retrieved)


def is_relevant(doc, relevant):
    """
    An indicator function to indicate whether the element doc
    is in the set of relevant docs

    :param doc: the document in question
    :param relevant: the set of relevant documents
    :return: 1 if the doc is in the set of relevant docs, 0
    otherwise
    """
    return 1 if doc in relevant else 0


def mean_average_precision(positives, candidates_ranked):
    """
    Computes the average of each of the precision values
    computed for the top  k documents for k = 1 to k = all

    :param positives: the set of similar documents
    :param candidates_ranked: a list of questions sorted
    in descending order by rank
    :return: the map, a real number from [0,1]
    """
    postives_as_set = set(positives)
    at_or_below_cutoff = set()
    total_precision = 0
    for candidate in candidates_ranked:
        at_or_below_cutoff.add(candidate)
        total_precision += precision(postives_as_set, at_or_below_cutoff) \
                           * is_relevant(candidate, postives_as_set)
    ave_precision = total_precision / len(postives_as_set)
    return ave_precision

def reciprocal_rank(positives, candidates_ranked):
    # find index of first occurence of one of positives in candidates_ranked
    positives = set(positives)
    index = 0
    while index < len(candidates_ranked):
        if candidates_ranked[index] in positives:
            break
        index += 1
    return 1.0 / (index + 1)

def precision_at_n(positives, candidates_ranked, n):
    """
    Computes the precision value of the top n ranked candidates
    against the set of relevant documents

    :param positives: the set of similar documents
    :param candidates_ranked: a list of questions sorted
    in descending order by rank
    :param n: consider the top n ranked questions
    :return: p@n, a real number from [0,1]
    """
    positives_as_set = set(positives)
    at_or_below_rank_n = set(candidates_ranked[:n+1])
    return precision(positives_as_set, at_or_below_rank_n)
