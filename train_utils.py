import random
from collections import namedtuple
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import helpers

NEGATIVE_QUERYS_PER_SAMPLE = 20
MAX_TITLE_LENGTH = 38
MAX_BODY_LENGTH = 72

Result = namedtuple("Result", "state_dict mrr")

def merge_title_and_body(corpus_entry):
    return np.hstack([corpus_entry.title, corpus_entry.body])

def pad(np_array, value, max_length):
    return np.pad(np_array, (0, max_length), 'constant', constant_values=value)[:max_length]

def process_batch_pairs(pairs, data, corpus, word_to_index):
    batch_querys_t = []
    batch_positives_t = []
    batch_negatives_t = []
    for query, positive in pairs:
        query_index_sequence = corpus[query].title
        batch_querys_t.append(pad(query_index_sequence, len(word_to_index), MAX_TITLE_LENGTH))
        positive_index_sequence = corpus[positive].title
        batch_positives_t.append(pad(positive_index_sequence, len(word_to_index), MAX_TITLE_LENGTH))
        negatives = [corpus[neg].title \
                    for neg in random.sample(data[(query, positive)], NEGATIVE_QUERYS_PER_SAMPLE)]
        negatives = [pad(neg, len(word_to_index), MAX_TITLE_LENGTH) for neg in negatives]
        batch_negatives_t.append(negatives)
    batch_querys_t = torch.from_numpy(np.array(batch_querys_t))
    batch_positives_t = torch.from_numpy(np.array(batch_positives_t))
    batch_negatives_t = torch.from_numpy(np.array(batch_negatives_t))

    batch_querys_b = []
    batch_positives_b = []
    batch_negatives_b = []
    for query, positive in pairs:
        query_index_sequence = corpus[query].body
        batch_querys_b.append(pad(query_index_sequence, len(word_to_index), MAX_BODY_LENGTH))
        positive_index_sequence = corpus[positive].body
        batch_positives_b.append(pad(positive_index_sequence, len(word_to_index), MAX_BODY_LENGTH))
        negatives = [corpus[neg].body \
                    for neg in random.sample(data[(query, positive)], NEGATIVE_QUERYS_PER_SAMPLE)]
        negatives = [pad(neg, len(word_to_index), MAX_BODY_LENGTH) for neg in negatives]
        batch_negatives_b.append(negatives)
    batch_querys_b = torch.from_numpy(np.array(batch_querys_b))
    batch_positives_b = torch.from_numpy(np.array(batch_positives_b))
    batch_negatives_b = torch.from_numpy(np.array(batch_negatives_b))

    batch_querys = [batch_querys_t, batch_querys_b]
    batch_positives = [batch_positives_t, batch_positives_b]
    batch_negatives = [batch_negatives_t, batch_negatives_b]
    return batch_querys, batch_positives, batch_negatives

def evaluate_model(model, data, corpus, word_to_index, cuda, eval_func):
    rrs = []
    for query in data.keys():
        positives, candidates = data[query]

        embeddings_t = []
        embeddings_b = []
        embeddings_t.append(pad(corpus[query].title, len(word_to_index), MAX_TITLE_LENGTH))
        embeddings_b.append(pad(corpus[query].body, len(word_to_index), MAX_BODY_LENGTH))
        for candidate in candidates:
            embeddings_t.append(pad(corpus[candidate].title, len(word_to_index), MAX_TITLE_LENGTH))
            embeddings_b.append(pad(corpus[candidate].body, len(word_to_index), MAX_BODY_LENGTH))
        embeddings_t = Variable(torch.from_numpy(np.array(embeddings_t)))
        embeddings_b = Variable(torch.from_numpy(np.array(embeddings_b)))
        if cuda:
            embeddings_t, embeddings_b = embeddings_t.cuda(), embeddings_b.cuda()

        encodings = (model(embeddings_t) + model(embeddings_b)) / 2
        similarities = F.cosine_similarity(encodings[1:],
                                           encodings[0].repeat(len(encodings)-1, 1), dim=1)
        _, candidates_ranked = zip(*sorted(zip(similarities.data, candidates), reverse=True))
        rrs.append(eval_func(positives, candidates_ranked))
    return np.mean(rrs)

def train_model(model, optimizer, criterion, train_data,\
                max_epochs, batch_size, cuda, eval_data=None):
    """
    Train the model with the given parameters.
    Returns the model at the epoch that produces the highest MRR
    on the dev set.
    """
    if cuda:
        model = model.cuda()

    models_by_epoch = []
    max_mrr = 0
    models_since_max_mrr = -1
    for epoch in tqdm(range(max_epochs)):
        model.train()
        similar_pairs = list(train_data.train.keys())
        random.shuffle(similar_pairs)

        losses = []
        for index, i in enumerate(tqdm(range(0, len(similar_pairs), batch_size))):
            query, positive, negatives = process_batch_pairs(similar_pairs[i:i + batch_size], \
                                            train_data.train, train_data.corpus, train_data.word_to_index)
            query_t, query_b = Variable(query[0]), Variable(query[1])
            positive_t, positive_b = Variable(positive[0]), Variable(positive[1])
            negatives_t, negatives_b = Variable(negatives[0]), Variable(negatives[1])
            if cuda:
                query_t, positive_t, negatives_t = \
                    query_t.cuda(), positive_t.cuda(), negatives_t.cuda()
                query_b, positive_b, negatives_b = \
                    query_b.cuda(), positive_b.cuda(), negatives_b.cuda()

            query_encoding = (model(query_t.long()) + model(query_b.long())) / 2
            positive_encoding = (model(positive_t.long()) + model(positive_b.long())) / 2
            # negative_encodings: (batch_sample_index, negative_query_index, seq_len)
            # input to model: (batch_sample_index, seq_len)
            # model output: (batch_sample_index, encoding_len)
            negative_encodings = torch.stack(\
                [(model(negatives_t[:, i].long())+model(negatives_b[:, i].long()))/2\
                for i in range(NEGATIVE_QUERYS_PER_SAMPLE)])
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
        if eval_data is None:
            mrr = evaluate_model(model, train_data.dev, train_data.corpus, train_data.word_to_index, cuda, helpers.reciprocal_rank)
        else:
            mrr = evaluate_model(model, eval_data.dev, eval_data.corpus, eval_data.word_to_index, cuda, helpers.reciprocal_rank)
        print(epoch, mrr)

        # Save the model for this epoch
        model.cpu()
        models_by_epoch.append(Result(model.state_dict(), mrr))
        if cuda:
            model.cuda()

        # Determine if we should stop training
        if mrr > max_mrr:
            max_mrr = mrr
            models_since_max_mrr = 0
        else:
            models_since_max_mrr += 1
        if models_since_max_mrr > 2:
            break

    # Pick the best epoch and return the model from that epoch
    best_state_dict = sorted(models_by_epoch, key=lambda x: x.mrr, reverse=True)[0].state_dict
    model.load_state_dict(best_state_dict)
    if eval_data is None:
        mrr = evaluate_model(model, train_data.dev, train_data.corpus, train_data.word_to_index, cuda, helpers.reciprocal_rank)
    else:
        mrr = evaluate_model(model, eval_data.dev, eval_data.corpus, eval_data.word_to_index, cuda, helpers.reciprocal_rank)
    print("best mrr", mrr)
    return model, max_mrr
