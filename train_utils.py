import random

import numpy as np
import torch
import helpers

def process_batch_pairs(pairs, data, corpus):
    batch_querys = []
    batch_positives = []
    batch_negatives = []
    for query, positive in pairs:
        batch_querys.append(corpus[query])
        batch_positives.append(corpus[positive])
        negatives = [corpus[neg] for neg in data[(query, positive)]]
        batch_negatives.append(negatives)

    batch_querys = torch.from_numpy(batch_querys)
    batch_positives = torch.from_numpy(batch_positives)
    batch_negatives = torch.from_numpy(batch_negatives)
    return batch_querys, batch_positives, batch_negatives

def train_model(model, optimizer, criterion, data,\
                max_epochs, batch_size, cuda):
    if cuda:
        model = model.cuda()

    model.train()
    for epoch in range(max_epochs):
        similar_pairs = list(data.train.keys())
        random.shuffle(similar_pairs)

        batches = [process_batch_pairs(similar_pairs[i:i + batch_size], data.train, data.corpus) \
                    for i in range(0, len(similar_pairs), batch_size)]
        for query, positive, negatives in batches:
            if cuda:
                query, positive, negatives = \
                    query.cuda(), positive.cuda(), negatives.cuda()

            query_encoding = model(query)
            positive_encoding = model(positive)
            negative_encodings = torch.stack(\
                                [model(negative) for negative in negatives])
            loss = criterion(positive_encoding, negative_encodings, query_encoding)

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
