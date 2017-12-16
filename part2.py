from collections import namedtuple
import itertools

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn.functional as F

import data_utils
import train_utils
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
# TOKENIZED_ANDROID_CORPUS = data_utils.load_tokenized_android_corpus()
# TOKENIZED_ANDROID_CORPUS = [
#     entry.title + entry.body for entry in TOKENIZED_ANDROID_CORPUS.values()
# ]
# TFIDF_VECTORIZER = TfidfVectorizer()
# TFIDF_VECTORS = TFIDF_VECTORIZER.fit_transform(TOKENIZED_ANDROID_CORPUS)
# QUERY_TO_INDEX = dict(zip(ANDROID_DATA.corpus.keys(), range(len(ANDROID_DATA.corpus))))
# AUC = helpers.evaluate_tfidf_auc(ANDROID_DATA.dev, TFIDF_VECTORS, QUERY_TO_INDEX)
# print("AUC", AUC)
# MAP = helpers.evaluate_tfidf(ANDROID_DATA.dev, TFIDF_VECTORS, QUERY_TO_INDEX, \
#                             helpers.mean_average_precision)
# MRR = helpers.evaluate_tfidf(ANDROID_DATA.dev, TFIDF_VECTORS, QUERY_TO_INDEX, \
#                             helpers.reciprocal_rank)
# PA1 = helpers.evaluate_tfidf(ANDROID_DATA.dev, TFIDF_VECTORS, QUERY_TO_INDEX, \
#                             lambda x,y: helpers.precision_at_n(x, y, 1))
# PA5 = helpers.evaluate_tfidf(ANDROID_DATA.dev, TFIDF_VECTORS, QUERY_TO_INDEX, \
#                             lambda x,y: helpers.precision_at_n(x, y, 5))
# print("MAP", MAP)
# print("MRR", MRR)
# print("PA1", PA1)
# print("PA5", PA5)

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
# while True:
#     LSTM_HYPERPARAMETERS = itertools.product(MARGINS, NUM_HIDDEN_UNITS, LEARNING_RATES)
#     for margin, num_hidden_units, learning_rate in LSTM_HYPERPARAMETERS:
#         model = LSTM(EMBEDDINGS, num_hidden_units, POOL_METHOD, CUDA)
#         criterion = helpers.MaxMarginLoss(margin)
#         parameters = filter(lambda p: p.requires_grad, model.parameters())
#         optimizer = torch.optim.Adam(parameters, lr=learning_rate)
#         model, mrr = train_utils.train_model(model, optimizer, criterion, ASK_UBUNTU_DATA, \
#                                         MAX_EPOCHS, BATCH_SIZE, CUDA, eval_data=ANDROID_DATA)
#         torch.save(model.state_dict(), "./models_part2/lstm_" +
#                                         str(margin) + "_" +
#                                         str(num_hidden_units) + "_" +
#                                         str(learning_rate) + "_" +
#                                         "auc=" + str(mrr))
#         MODELS.append((mrr, margin, num_hidden_units, learning_rate))

##############################################################################
# Train models by adverserial domain adaptation and evaluate
##############################################################################
# @782
MAX_EPOCHS = 50
BATCH_SIZE = 32
MARGINS = [0.2]
FILTER_WIDTH = 2
POOL_METHOD = "average"
FEATURE_DIM = 240
DIS_NUM_HIDDEN_UNITS = [150, 200]
DIS_LEARNING_RATES = [-1E-3]
ENC_LEARNING_RATE = 1E-3
DIS_TRADE_OFF_RATES = [1E-7, 1E-8, 1E-9]
# ENCODER_MODELS = [LSTM(EMBEDDINGS, num_hidden_units, POOL_METHOD, CUDA)]

while True:
    DIS_HYPERPARAMETERS = itertools.product(DIS_LEARNING_RATES, DIS_NUM_HIDDEN_UNITS, DIS_TRADE_OFF_RATES, MARGINS)

    for dis_lr, dis_hidden_units, trade_off, margin in DIS_HYPERPARAMETERS:
        enc_model = LSTM(EMBEDDINGS, FEATURE_DIM, POOL_METHOD, CUDA)
        dis_model = BinaryClassifier(FEATURE_DIM, dis_hidden_units)
        model, auc = part2_train_utils.train_model(
            enc_model,
            dis_model,
            trade_off,
            ASK_UBUNTU_DATA,
            ANDROID_DATA,
            MAX_EPOCHS,
            BATCH_SIZE,
            ENC_LEARNING_RATE,
            dis_lr,
            margin,
            CUDA,
        )
        print("max auc", auc)
        torch.save(model.state_dict(), "./models_part2/lstm_" +\
                                        str(margin) + "_" +\
                                        str(dis_hidden_units) + "_" +\
                                        str(trade_off) + "_" +\
                                        "auc=" + str(auc))
        print("should have saved", "./models_part2/lstm_" +\
                                        str(margin) + "_" +\
                                        str(dis_hidden_units) + "_" +\
                                        str(trade_off) + "_" +\
                                        "auc=" + str(auc))
