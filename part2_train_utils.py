import random
from collections import namedtuple
import numpy as np

from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import train_utils
from helpers import MaxMarginLoss as MML
from helpers import pad
from helpers import merge_title_and_body
from helpers import NEGATIVE_QUERYS_PER_SAMPLE
import helpers
from meter import AUCMeter

Result = namedtuple("Result", "state_dict auc")
TrainerPack = namedtuple("TrainerPack", "model optimizer criterion")
SOURCE_LABEL = 0
TARGET_LABEL = 1
IS_SIMMILAR_LABEL = 1
NOT_SIMMILAR_LABEL = 0
NEGATIVE_QUERYS_PER_SAMPLE = train_utils.NEGATIVE_QUERYS_PER_SAMPLE
TARGET_QUERIES_PER_SAMPLE = NEGATIVE_QUERYS_PER_SAMPLE
SOURCE_QUERIES_PER_SAMPLE = NEGATIVE_QUERYS_PER_SAMPLE
MAXIMUM_FALSE_POSITIVE_RATIO = 0.05

def discriminator_model_loss(dis_model, target_qs, source_qs, cuda):
    all_qs = torch.cat((target_qs, source_qs), dim=0)
    input = torch.stack([dis_model(all_qs[i]) for i in range(all_qs.data.shape[0])], dim=0)
    labels = Variable(torch.from_numpy(
        np.array([SOURCE_LABEL]*source_qs.data.shape[0] + [TARGET_LABEL]*target_qs.data.shape[0])
    ))
    labels = labels.cuda() if cuda else labels
    return F.nll_loss(input, labels)

def evaluate_model(model, data, corpus, word_to_index, cuda):
    auc = AUCMeter()
    for query in data.keys():
        positives = set(data[query][0])
        candidates = data[query][1]

        embeddings = [pad(merge_title_and_body(corpus[query]), len(word_to_index))]
        targets = []
        for candidate in candidates:
            embeddings.append(pad(merge_title_and_body(corpus[candidate]), len(word_to_index)))
            targets.append(IS_SIMMILAR_LABEL if candidate in positives else NOT_SIMMILAR_LABEL)
        embeddings = Variable(torch.from_numpy(np.array(embeddings)))
        targets = torch.from_numpy(np.array(targets))
        if cuda:
            embeddings = embeddings.cuda()

        encodings = model(embeddings)
        query_encoding = encodings[0]
        candidate_encodings = encodings[1:]
        similarities = (F.cosine_similarity(candidate_encodings, query_encoding.repeat(len(encodings)-1, 1), dim=1))
        auc.add(similarities.data, targets)
    return auc.value(MAXIMUM_FALSE_POSITIVE_RATIO)

def get_dis_batch(batch_size, corpus, word_to_index):
    keys = list(corpus.keys())
    random.shuffle(keys)
    keys = keys[:batch_size]
    query_index_sequences = [merge_title_and_body(corpus[k]) for k in keys]
    embedded = [pad(seq, len(word_to_index)) for seq in query_index_sequences]
    batch = torch.from_numpy(np.array(embedded))
    return batch

def train_model(enc_model, dis_model, lambda_tradeoff, source_data, target_data,
                max_epochs, batch_size, enc_lr, dis_lr, margin, cuda):
    """
    Train the model with the given parameters.
    Returns the model at the epoch that produces the highest AUC
    on the dev set.
    """
    criterion = MML(margin)
    dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=dis_lr)
    parameters = filter(lambda p: p.requires_grad, enc_model.parameters())
    enc_optimizer = torch.optim.Adam(parameters, lr=enc_lr)

    if cuda:
        enc_model = enc_model.cuda()
        dis_model = dis_model.cuda()

    models_by_epoch = []
    max_auc = 0
    models_since_max_auc = -1

    for epoch in tqdm(range(max_epochs)):
        enc_model.train()
        dis_model.train()
        similar_pairs = list(source_data.train.keys())
        random.shuffle(similar_pairs)

        dis_losses = []
        enc_losses = []
        for index, i in enumerate(tqdm(range(0, len(similar_pairs), batch_size))):
            query, positive, negatives = train_utils.process_batch_pairs(
                similar_pairs[i:i + batch_size],
                source_data.train,
                source_data.corpus,
                source_data.word_to_index
            )
            ubuntus = get_dis_batch(
                SOURCE_QUERIES_PER_SAMPLE,
                source_data.corpus,
                source_data.word_to_index
            )
            androids = get_dis_batch(
                TARGET_QUERIES_PER_SAMPLE,
                target_data.corpus,
                target_data.word_to_index
            )

            query = Variable(query)
            positive = Variable(positive)
            negatives = Variable(negatives)
            ubuntus = Variable(ubuntus)
            androids = Variable(androids)
            if cuda:
                query = query.cuda()
                positive = positive.cuda()
                negatives = negatives.cuda()
                ubuntus = ubuntus.cuda()
                androids = androids.cuda()

            query_encoding = enc_model(query.long())
            positive_encoding = enc_model(positive.long())
            negative_encodings = torch.stack(
                [enc_model(negatives[:, i].long()) for i in range(NEGATIVE_QUERYS_PER_SAMPLE)]
            )
            ubuntu_encodings = enc_model(ubuntus.long())
            android_encodings = enc_model(androids.long())

            dis_loss = discriminator_model_loss(dis_model, android_encodings, ubuntu_encodings, cuda)
            enc_loss = criterion(positive_encoding, negative_encodings, query_encoding)

            total_loss = enc_loss - lambda_tradeoff*dis_loss

            dis_optimizer.zero_grad()
            enc_optimizer.zero_grad()
            total_loss.backward()
            dis_optimizer.step()
            enc_optimizer.step()
            dis_losses.append(dis_loss.data)
            enc_losses.append(enc_loss.data)

        # Evaluate on the dev set and save the AUC and model parameters
        enc_model.eval()
        mrr = train_utils.evaluate_model(enc_model, source_data.dev, source_data.corpus, source_data.word_to_index, cuda, helpers.reciprocal_rank)
        print(epoch, "mrr source", mrr)
        mrr = train_utils.evaluate_model(enc_model, target_data.dev, target_data.corpus, target_data.word_to_index, cuda, helpers.reciprocal_rank)
        print(epoch, "mrr target", mrr)
        auc = evaluate_model(enc_model, source_data.dev, source_data.corpus, source_data.word_to_index, cuda)
        print(epoch, "auc source", auc)
        auc = evaluate_model(enc_model, target_data.dev, target_data.corpus, target_data.word_to_index, cuda)
        print(epoch, "auc target", auc)
        models_by_epoch.append(Result(enc_model.state_dict(), auc))
        if auc > max_auc:
            max_auc = auc
            models_since_max_auc = 0
        else:
            models_since_max_auc += 1
        if models_since_max_auc > 2:
            break
    # Pick the best epoch and return the model from that epoch
    # best_state_dict = sorted(models_by_epoch, key=lambda x: x.auc, reverse=True)[0].state_dict
    # enc_model.load_state_dict(best_state_dict)
    return enc_model, max_auc