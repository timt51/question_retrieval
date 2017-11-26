import random

import numpy as np
import torch
import helpers

def merge_title_and_body(corpus_entry):
    return np.hstack([corpus_entry.title, corpus_entry.body])

def pad(np_array, max_length, value):
    return np.pad(np_array, (0, max_length-len(np_array)), 'constant', constant_values=value)

def process_batch_pairs(pairs, data, corpus, word_to_index):
    max_length = 0
    for query, positive in pairs:
        query = merge_title_and_body(corpus[query])
        max_length = max(max_length, len(query))
        positive = merge_title_and_body(corpus[positive])
        max_length = max(max_length, positive)
        negatives = [merge_title_and_body(corpus[neg]) for neg in data[(query, positive)]]
        max_length = max(max_length, max(map(len, negatives)))

    batch_querys = []
    batch_positives = []
    batch_negatives = []
    for query, positive in pairs:
        query = merge_title_and_body(corpus[query])
        batch_querys.append(pad(query, max_length, len(word_to_index)))
        positive = merge_title_and_body(corpus[positive])
        batch_positives.append(pad(positive, max_length, len(word_to_index)))
        negatives = [merge_title_and_body(corpus[neg]) for neg in data[(query, positive)]]
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
    for epoch in range(max_epochs):
        similar_pairs = list(data.train.keys())
        random.shuffle(similar_pairs)

        batches = [process_batch_pairs(similar_pairs[i:i + batch_size], \
                    data.train, data.corpus, data.word_to_index) \
                    for i in range(0, len(similar_pairs), batch_size)]
        for query, positive, negatives in batches:
            if cuda:
                query, positive, negatives = \
                    query.cuda(), positive.cuda(), negatives.cuda()

            query_encoding = model(query)
            positive_encoding = model(positive)
            negative_encodings = torch.stack(\
                                [model(negatives[:, i]) for i in range(len(negatives))])
            loss = criterion(positive_encoding, negative_encodings, query_encoding)

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
        # Evaluate on the dev set and save the MRR and model parameters
    # Pick the best epoch and return the model from that epoch
