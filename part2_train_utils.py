import random
from collections import namedtuple

from tqdm import tqdm
import torch
from torch.autograd import Variable

from train_utils import process_batch_pairs, NEGATIVE_QUERYS_PER_SAMPLE

Result = namedtuple("Result", "state_dict auc")
TrainerPack = namedtuple("TrainerPack", "model optimizer criterion")


def evaluate_model(model, data, corpus, word_to_index, cuda):
    pass

def discriminator_model_loss(negative_encodings, android_encodings):
    # TODO: Validate the assumption that negative encodings are
    pass

def encoder_model_loss():
    pass

def train_model(enc_model, dis_model, lambda_tradeoff, source_data, target_data, max_epochs, batch_size,cuda):
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
        andriod_queries = list(target_data.corpus.values())
        random.shuffle(andriod_queries)

        enc_losses = []
        for index, i in enumerate(tqdm(range(0, len(similar_pairs), batch_size))):
            query, positive, negatives = process_batch_pairs(
                similar_pairs[i:i + batch_size],
                source_data.train,
                source_data.corpus,
                source_data.word_to_index
            )
            query, positive, negatives = Variable(query), Variable(positive), Variable(negatives)
            if cuda:
                query, positive, negatives = query.cuda(), positive.cuda(), negatives.cuda()

            query_encoding = enc_model(query.long())
            positive_encoding = enc_model(positive.long())
            negative_encodings = torch.stack(
                [enc_model(negatives[:, i].long()) for i in range(NEGATIVE_QUERYS_PER_SAMPLE)]
            )
            android_encodings = torch.stack(
                [enc_model(andriod_queries.pop()) for _ in range(NEGATIVE_QUERYS_PER_SAMPLE)]
            )

            dis_loss = discriminator_model_loss(negative_encodings, android_encodings)
            enc_loss = encoder_model_loss(dis_loss, lambda_tradeoff)

            dis_optimizer.zero_grad()
            enc_optimizer.zero_grad()
            dis_loss.backward()
            enc_loss.backward()
            dis_optimizer.step()
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