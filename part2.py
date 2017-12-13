from collections import namedtuple
import itertools

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn.functional as F

import data_utils
from models import BinaryClassifier
import part2_train_utils
import helpers

##############################################################################
# Settings
##############################################################################
CUDA = True

##############################################################################
# Load the dataset
##############################################################################
Data = namedtuple("Data", \
        "corpus train dev test embeddings word_to_index")

data_utils.download_ask_ubuntu_dataset()
EMBEDDINGS, WORD_TO_INDEX = data_utils.load_embeddings()
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
                                # TOKENIZED_ANDROID_CORPUS = data_utils.load_tokenized_android_corpus(WORD_TO_INDEX)
                                # TOKENIZED_ANDROID_CORPUS = [
                                #     entry.title + entry.body for entry in TOKENIZED_ANDROID_CORPUS.values()
                                # ]
                                # TFIDF_VECTORIZER = TfidfVectorizer()
                                # TFIDF_VECTORS = TFIDF_VECTORIZER.fit_transform(TOKENIZED_ANDROID_CORPUS)

##############################################################################
# Train and evaluate the models for Part 2
##############################################################################
RESULTS = []
MARGIN = ["""TODO"""]
MAX_EPOCHS = 50
BATCH_SIZE = 64
FILTER_WIDTH = 2
POOL_METHOD = "average"
FEATURE_DIM = 300
DIS_NUM_HIDDEN_UNITS = ["""TODO"""]
DIS_LEARNING_RATES = ["""TODO"""]
DIS_TRADE_OFF_RATES = ["""TODO"""]
ENCODER_MODELS = ["""TODO"""]

DIS_HYPERPARAMETERS = itertools.product(DIS_LEARNING_RATES, DIS_NUM_HIDDEN_UNITS, DIS_TRADE_OFF_RATES, ENCODER_MODELS)

for learning_rate, num_hidden_units, trade_off, enc_model in DIS_HYPERPARAMETERS:
    dis_model = BinaryClassifier(EMBEDDINGS, FEATURE_DIM, num_hidden_units)
    model, mrr = part2_train_utils.train_model(
        enc_model,
        dis_model,
        trade_off,
        ASK_UBUNTU_DATA,
        ANDROID_DATA,
        MAX_EPOCHS,
        BATCH_SIZE,
        CUDA
    )
