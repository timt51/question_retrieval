import random
from collections import namedtuple
import numpy as np

from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from train_utils import process_batch_pairs, NEGATIVE_QUERYS_PER_SAMPLE, pad, merge_title_and_body
from helpers import MaxMarginLoss as MML
from meter import AUCMeter

Result = namedtuple("Result", "state_dict auc")
TrainerPack = namedtuple("TrainerPack", "model optimizer criterion")
SOURCE_LABEL = 0
TARGET_LABEL = 1
IS_SIMMILAR_LABEL = 1
NOT_SIMMILAR_LABEL = 0
TARGET_QUERIES_PER_SAMPLE = NEGATIVE_QUERYS_PER_SAMPLE
SOURCE_QUERIES_PER_SAMPLE = NEGATIVE_QUERYS_PER_SAMPLE
MAXIMUM_FALSE_POSITIVE_RATIO = 0.05

def discriminator_model_loss(dis_model, target_qs, source_qs):
    all_qs = torch.cat((target_qs, source_qs), dim=0)
    input = torch.stack([dis_model(all_qs[:, i]) for i in range(all_qs.data.shape[1])], dim=1)
    labels = torch.from_numpy(
        np.array([SOURCE_LABEL]*source_qs.data.shape[0] + [TARGET_LABEL]*target_qs.data.shape[0])
    )
    return F.nll_loss(input, labels)


def evaluate_model(model, data, corpus, word_to_index, cuda):
    auc = AUCMeter()
    for query in data.keys():
        positives = set(data[query][0])
        candidates = data[query][1]

        embeddings = [pad(merge_title_and_body(corpus[query]), len(word_to_index))]
        targets = [IS_SIMMILAR_LABEL]
        for candidate in candidates:
            embeddings.append(pad(merge_title_and_body(corpus[candidate]), len(word_to_index)))
            targets.append(IS_SIMMILAR_LABEL if candidate in positives else NOT_SIMMILAR_LABEL)
        embeddings = Variable(torch.from_numpy(np.array(embeddings)))
        if cuda:
            embeddings = embeddings.cuda()

        encodings = model(embeddings)
        query_encoding = encodings[0]
        candidate_encodings = encodings[1:]
        similarities = (F.cosine_similarity(candidate_encodings, query_encoding.repeat(len(encodings)-1, 1), dim=1))
        auc.add(similarities.data, targets)
    return auc.value(MAXIMUM_FALSE_POSITIVE_RATIO)


def get_android_batch(batch_size, corpus, word_to_index):
    embedded = []
    for _ in range(batch_size):
        query_index_sequences = [merge_title_and_body(corpus.pop()) for _ in range(TARGET_QUERIES_PER_SAMPLE)]
        embedded.append([pad(seq, len(word_to_index)) for seq in query_index_sequences])
    android_batch = torch.from_numpy(np.array(embedded))
    return android_batch

def train_model(enc_model, dis_model, lambda_tradeoff, source_data, target_data,
                max_epochs, batch_size, enc_lr, dis_lr, cuda):
    """
    Train the model with the given parameters.
    Returns the model at the epoch that produces the highest AUC
    on the dev set.
    """
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
        android_queries = list(target_data.corpus.values())
        random.shuffle(android_queries)

        enc_losses = []
        for _, i in enumerate(tqdm(range(0, len(similar_pairs), batch_size))):
            query, positive, negatives = process_batch_pairs(
                similar_pairs[i:i + batch_size],
                source_data.train,
                source_data.corpus,
                source_data.word_to_index
            )
            androids = get_android_batch(
                batch_size,
                android_queries,
                target_data.word_to_index
            )

            query = Variable(query)
            positive = Variable(positive)
            negatives = Variable(negatives)
            androids = Variable(androids)
            if cuda:
                query = query.cuda()
                positive = positive.cuda()
                negatives = negatives.cuda()
                androids = androids.cuda()

            query_encoding = enc_model(query.long())
            positive_encoding = enc_model(positive.long())
            negative_encodings = torch.stack(
                [enc_model(negatives[:, i].long()) for i in range(NEGATIVE_QUERYS_PER_SAMPLE)]
            )
            android_encodings = torch.stack(
                [enc_model(androids[:, i].long()) for i in range(TARGET_QUERIES_PER_SAMPLE)]
            )

            dis_loss = discriminator_model_loss(dis_model, android_encodings, negative_encodings)
            enc_loss = MML(enc_model).forward(positive_encoding, negative_encodings, query_encoding)\
                       - lambda_tradeoff*dis_loss

            dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=dis_lr)
            dis_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()

            enc_optimizer = torch.optim.Adam(enc_model.parameters(), lr=enc_lr)
            enc_optimizer.zero_grad()
            enc_loss.backward()
            enc_optimizer.step()

            enc_losses.append(enc_loss.data)

        # Evaluate on the dev set and save the AUC and model parameters
        enc_model.eval()
        auc = evaluate_model(enc_model, target_data.dev, target_data.corpus, target_data.word_to_index, cuda)
        print(epoch, auc)
        models_by_epoch.append(Result(enc_model.state_dict(), auc))
        if auc > max_auc:
            max_auc = auc
            models_since_max_auc = 0
        else:
            models_since_max_auc += 1
        if models_since_max_auc > 2:
            break
    # Pick the best epoch and return the model from that epoch
    best_state_dict = sorted(models_by_epoch, key=lambda x: x.auc, reverse=True)[0].state_dict
    enc_model.load_state_dict(best_state_dict)
    return enc_model, max_auc