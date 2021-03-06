import random
from collections import namedtuple
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import helpers
from helpers import pad
from helpers import merge_title_and_body
from helpers import NEGATIVE_QUERYS_PER_SAMPLE
import part2_train_utils

Result = namedtuple("Result", "state_dict mrr")

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

def evaluate_model(model, data, corpus, word_to_index, cuda, eval_func):
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

            losses.append(loss.data)

        # Evaluate on the dev set and save the MRR and model parameters
        model.eval()
        mrr_source = evaluate_model(model, train_data.dev, train_data.corpus, train_data.word_to_index, cuda, helpers.reciprocal_rank)
        auc_source = part2_train_utils.evaluate_model(model, train_data.dev, train_data.corpus, train_data.word_to_index, cuda)
        print(epoch, "mrr source", mrr_source)
        print(epoch, "auc source", auc_source)
        if eval_data is not None:
            mrr_target = evaluate_model(model, eval_data.dev, eval_data.corpus, eval_data.word_to_index, cuda, helpers.reciprocal_rank)
            auc_target = part2_train_utils.evaluate_model(model, eval_data.dev, eval_data.corpus, eval_data.word_to_index, cuda)
            print(epoch, "mrr target", mrr_target)
            print(epoch, "auc target", auc_target)
            mrr_source = auc_target
        print(mrr_source)
        # Save the model for this epoch
        model.cpu()
        models_by_epoch.append(Result(model.state_dict(), mrr_source))
        if cuda:
            model.cuda()

        # Determine if we should stop training
        if mrr_source > max_mrr:
            max_mrr = mrr_source
            models_since_max_mrr = 0
        else:
            models_since_max_mrr += 1
        if models_since_max_mrr > 2:
            break

    # Pick the best epoch and return the model from that epoch
    best_state_dict = sorted(models_by_epoch, key=lambda x: x.mrr, reverse=True)[0].state_dict
    model.load_state_dict(best_state_dict)
    print("best mrr", max_mrr)
    return model, max_mrr
