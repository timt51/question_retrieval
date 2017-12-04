import random
from collections import namedtuple

from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import helpers

NEGATIVE_QUERYS_PER_SAMPLE = 20
MAX_LENGTH = 100

Result = namedtuple("Result", "state_dict mrr")

def merge_title_and_body(corpus_entry):
    return np.hstack([corpus_entry.title, corpus_entry.body])

def pad(np_array, value):
    return np.pad(np_array, (0, MAX_LENGTH), 'constant', constant_values=value)[:MAX_LENGTH]

def process_batch_pairs(pairs, data, corpus, word_to_index):
    batch_querys = []
    batch_positives = []
    batch_negatives = []
    for query, positive in pairs:
        query_index_sequence = merge_title_and_body(corpus[query])
        batch_querys.append(pad(query_index_sequence, len(word_to_index)))
        positive_index_sequence = merge_title_and_body(corpus[positive])
        batch_positives.append(pad(positive_index_sequence, len(word_to_index)))
        negatives = [merge_title_and_body(corpus[neg]) \
                    for neg in random.sample(data[(query, positive)], NEGATIVE_QUERYS_PER_SAMPLE)]
        negatives = [pad(neg, len(word_to_index)) for neg in negatives]
        batch_negatives.append(negatives)

    batch_querys = torch.from_numpy(np.array(batch_querys))
    batch_positives = torch.from_numpy(np.array(batch_positives))
    batch_negatives = torch.from_numpy(np.array(batch_negatives))
    return batch_querys, batch_positives, batch_negatives

def evaluate_model(model, data, corpus, word_to_index, cuda):
    rrs = []
    for query in data.keys():
        positives, candidates = data[query]

        embeddings = []
        embeddings.append(pad(merge_title_and_body(corpus[query]), len(word_to_index)))
        for candidate in candidates:
            embeddings.append(pad(merge_title_and_body(corpus[candidate]), len(word_to_index)))
        embeddings = Variable(torch.from_numpy(np.array(embeddings)))
        if cuda:
            embeddings = embeddings.cuda()

        encodings = model(embeddings)
        similarities = F.cosine_similarity(encodings[1:],
                                           encodings[0].repeat(len(encodings)-1, 1), dim=1)
        _, candidates_ranked = zip(*sorted(zip(similarities.data, candidates), reverse=True))
        rrs.append(helpers.reciprocal_rank(positives, candidates_ranked))
    return np.mean(rrs)

def train_model(model, optimizer, criterion, data,\
                max_epochs, batch_size, cuda):
    """
    Train the model with the given parameters.
    Returns the model at the epoch that produces the highest MRR
    on the dev set.
    """
    if cuda:
        model = model.cuda()

    models_by_epoch = []
    for epoch in tqdm(range(max_epochs)):
        model.train()
        similar_pairs = list(data.train.keys())
        random.shuffle(similar_pairs)

        losses = []
        for index, i in enumerate(tqdm(range(0, len(similar_pairs), batch_size))):
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if index % 150 == 149:
                print("Average loss: " + str(np.mean(losses)))
                losses = []
            losses.append(loss.data)

        # Evaluate on the dev set and save the MRR and model parameters
        model.eval()
        mrr = evaluate_model(model, data.dev, data.corpus, data.word_to_index, cuda)
        print(epoch, mrr)
        models_by_epoch.append(Result(model.state_dict(), mrr))
    # Pick the best epoch and return the model from that epoch
    best_state_dict = sorted(models_by_epoch, key=lambda x: x.mrr, reverse=True)[0]
    model.load_state_dict(best_state_dict)
    return model
