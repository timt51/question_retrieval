from collections import namedtuple
import itertools

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
<<<<<<< HEAD
from sklearn.metrics.pairwise import cosine_similarity as cosine
=======
import torch.nn.functional as F
>>>>>>> part2-model

import data_utils
from models import BinaryClassifier, LSTM, CNN
import part2_train_utils
import helpers

##############################################################################
# Settings
##############################################################################
CUDA = True

##############################################################################
# Load the dataset
##############################################################################
Data = namedtuple("Data", "corpus train dev test embeddings word_to_index")

data_utils.download_ask_ubuntu_dataset()
EMBEDDINGS, WORD_TO_INDEX = data_utils.load_part2_embeddings()
ASK_UBUNTU_CORPUS = data_utils.load_corpus(WORD_TO_INDEX)
ASK_UBUNTU_TRAIN_DATA = data_utils.load_train_data()
ASK_UBUNTU_DEV_DATA, ASK_UBUNTU_TEST_DATA = data_utils.load_eval_data()
ASK_UBUNTU_DATA = Data(ASK_UBUNTU_CORPUS, ASK_UBUNTU_TRAIN_DATA,\
                        ASK_UBUNTU_DEV_DATA, ASK_UBUNTU_TEST_DATA,\
                        EMBEDDINGS, WORD_TO_INDEX)

data_utils.download_android_dataset()
ANDROID_CORPUS = data_utils.load_android_corpus(WORD_TO_INDEX)
ANDROID_DEV_DATA, ANDROID_TEST_DATA = data_utils.load_android_eval_data()
ANDROID_DATA = Data(ANDROID_CORPUS, None,\
                      ANDROID_DEV_DATA, ANDROID_TEST_DATA,\
                      EMBEDDINGS, WORD_TO_INDEX)

##############################################################################
# Train and evaluate a baseline TFIDF model
##############################################################################
def evaluate_tfidf(data, tfidf_vectors, query_to_index, eval_func):
    rrs = []
    for entry_id, eval_query_result in data.items():
        similar_ids = eval_query_result.similar_ids
        candidate_ids = eval_query_result.candidate_ids

        entry_encoding = tfidf_vectors[query_to_index[entry_id]]
        candidate_similarities = []
        for candidate_id in candidate_ids:
            candidate_encoding = tfidf_vectors[query_to_index[candidate_id]]
            similarity = cosine(entry_encoding, candidate_encoding)
            candidate_similarities.append((candidate_id, similarity))
        ranked_candidates = sorted(candidate_similarities, key=lambda x: x[1], reverse=True)
        ranked_candidates = [x[0] for x in ranked_candidates]
        rrs.append(eval_func(similar_ids, ranked_candidates))
    return np.mean(rrs)
TOKENIZED_ANDROID_CORPUS = data_utils.load_tokenized_android_corpus()
TOKENIZED_ANDROID_CORPUS = [
    entry.title + entry.body for entry in TOKENIZED_ANDROID_CORPUS.values()
]
TFIDF_VECTORIZER = TfidfVectorizer()
TFIDF_VECTORS = TFIDF_VECTORIZER.fit_transform(TOKENIZED_ANDROID_CORPUS)
QUERY_TO_INDEX = dict(zip(ANDROID_DATA.corpus.keys(), range(len(ANDROID_DATA.corpus))))
MAP = evaluate_tfidf(ANDROID_DATA.dev, TFIDF_VECTORS, QUERY_TO_INDEX, helpers.mean_average_precision)
MRR = evaluate_tfidf(ANDROID_DATA.dev, TFIDF_VECTORS, QUERY_TO_INDEX, helpers.reciprocal_rank)
PA1 = evaluate_tfidf(ANDROID_DATA.dev, TFIDF_VECTORS, QUERY_TO_INDEX, lambda x,y: helpers.precision_at_n(x, y, 1))
PA5 = evaluate_tfidf(ANDROID_DATA.dev, TFIDF_VECTORS, QUERY_TO_INDEX, lambda x,y: helpers.precision_at_n(x, y, 5))
print("MAP", MAP)
print("MRR", MRR)
print("PA1", PA1)
print("PA5", PA5)

##############################################################################
# Train models by direct transfer and evaluate
##############################################################################
# RESULTS = []
# MARGINS = [0.2]
# MAX_EPOCHS = 50
# BATCH_SIZE = 32
# FILTER_WIDTHS = [3]
# POOL_METHOD = "average"
# FEATURE_DIMS = [667]
# DROPOUT_PS = [0.1]
# NUM_HIDDEN_UNITS = [240]
# LEARNING_RATES = [1E-3]
# MODELS = []
# ##############################################################################
# LSTM_HYPERPARAMETERS = itertools.product(MARGINS, NUM_HIDDEN_UNITS, LEARNING_RATES)
# for margin, num_hidden_units, learning_rate in LSTM_HYPERPARAMETERS:
#     model = models.LSTM(EMBEDDINGS, num_hidden_units, POOL_METHOD, CUDA)
#     criterion = helpers.MaxMarginLoss(margin)
#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = torch.optim.Adam(parameters, lr=learning_rate)
#     model, mrr = train_utils.train_model(model, optimizer, criterion, ASK_UBUNTU_DATA, \
#                                     MAX_EPOCHS, BATCH_SIZE, CUDA, eval_data=ANDROID_DATA)
#     torch.save(model.state_dict(), "./models_part2/lstm_" +
#                                     str(margin) + "_" +
#                                     str(num_hidden_units) + "_" +
#                                     str(learning_rate))
#     MODELS.append((mrr, margin, num_hidden_units, learning_rate))

##############################################################################
# Train models by adverserial domain adaptation and evaluate
##############################################################################
# MAX_EPOCHS = 50
# BATCH_SIZE = 32
# FILTER_WIDTH = 2
# POOL_METHOD = "average"
# FEATURE_DIM = 240
# DIS_NUM_HIDDEN_UNITS = [100]
# DIS_LEARNING_RATES = [-1E-3]
# ENC_LEARNING_RATE = 1E-3
# DIS_TRADE_OFF_RATES = [1E-6]
# # ENCODER_MODELS = [LSTM(EMBEDDINGS, num_hidden_units, POOL_METHOD, CUDA)]

# DIS_HYPERPARAMETERS = itertools.product(DIS_LEARNING_RATES, DIS_NUM_HIDDEN_UNITS, DIS_TRADE_OFF_RATES)

# for dis_lr, dis_hidden_units, trade_off in DIS_HYPERPARAMETERS:
#     enc_model = LSTM(EMBEDDINGS, FEATURE_DIM, POOL_METHOD, CUDA)
#     dis_model = BinaryClassifier(FEATURE_DIM, dis_hidden_units)
#     model, mrr = part2_train_utils.train_model(
#         enc_model,
#         dis_model,
#         trade_off,
#         ASK_UBUNTU_DATA,
#         ANDROID_DATA,
#         MAX_EPOCHS,
#         BATCH_SIZE,
#         ENC_LEARNING_RATE,
#         dis_lr,
#         CUDA,
#     )
