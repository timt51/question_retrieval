import random

from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable

import helpers

NEGATIVE_QUERYS_PER_SAMPLE = 2

def merge_title_and_body(corpus_entry):
    return np.hstack([corpus_entry.title, corpus_entry.body])

def pad(np_array, max_length, value):
    return np.pad(np_array, (0, max_length-len(np_array)), 'constant', constant_values=value)

def process_batch_pairs(pairs, data, corpus, word_to_index):
    max_length = 0
    for query, positive in pairs:
        query_index_sequence = merge_title_and_body(corpus[query])
        max_length = max(max_length, len(query_index_sequence))
        positive_index_sequence = merge_title_and_body(corpus[positive])
        max_length = max(max_length, len(positive_index_sequence))
        negatives = [merge_title_and_body(corpus[neg]) for neg in data[(query, positive)]]
        max_length = max(max_length, max(map(len, negatives)))

    batch_querys = []
    batch_positives = []
    batch_negatives = []
    for query, positive in pairs:
        query_index_sequence = merge_title_and_body(corpus[query])
        batch_querys.append(pad(query_index_sequence, max_length, len(word_to_index)))
        positive_index_sequence = merge_title_and_body(corpus[positive])
        batch_positives.append(pad(positive_index_sequence, max_length, len(word_to_index)))
        negatives = [merge_title_and_body(corpus[neg]) \
                    for neg in random.sample(data[(query, positive)], NEGATIVE_QUERYS_PER_SAMPLE)]
        negatives = [pad(neg, max_length, len(word_to_index)) for neg in negatives]
        batch_negatives.append(negatives)

    batch_querys = torch.from_numpy(np.array(batch_querys))
    batch_positives = torch.from_numpy(np.array(batch_positives))
    batch_negatives = torch.from_numpy(np.array(batch_negatives))
    return batch_querys, batch_positives, batch_negatives

def train_model(model, optimizer, criterion, data,\
                max_epochs, batch_size, cuda):
    """
    Train the model with the given parameters.
    Returns the model at the epoch that produces the highest MRR
    on the dev set.
    """
    if cuda:
        model = model.cuda()

    model.train()
    for epoch in tqdm(range(max_epochs)):
        similar_pairs = list(data.train.keys())
        random.shuffle(similar_pairs)

        for i in tqdm(range(0, len(similar_pairs), batch_size)):
            query, positive, negatives = process_batch_pairs(similar_pairs[i:i + batch_size], \
                                            data.train, data.corpus, data.word_to_index)
            query, positive, negatives = Variable(query), Variable(positive), Variable(negatives)
            if cuda:
                query, positive, negatives = \
                    query.cuda(), positive.cuda(), negatives.cuda()

            query_encoding = model(query.long())
            positive_encoding = model(positive.long())
            # negative_encodings: (batch_sample_index, negative_query_index, seq_len)
            # input to model: (batch_sample_index, seq_len)
            # model output: (batch_sample_index, encoding_len)
            negative_encodings = torch.stack(\
                [model(negatives[:, i].long()) for i in range(NEGATIVE_QUERYS_PER_SAMPLE)])
            # negative_encodings: (negative_query_index, batch_sample_index, encoding_len)
            loss = criterion(positive_encoding, negative_encodings, query_encoding)

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
        # Evaluate on the dev set and save the MRR and model parameters
    # Pick the best epoch and return the model from that epoch
    return 1
