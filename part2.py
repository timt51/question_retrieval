from collections import namedtuple

import torch
from sklearn.feature_extraction.text import TfidfVectorizer

import data_utils
import models
import train_utils
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
TOKENIZED_ANDROID_CORPUS = data_utils.load_tokenized_android_corpus(WORD_TO_INDEX)
TOKENIZED_ANDROID_CORPUS = [
    entry.title + entry.body for entry in TOKENIZED_ANDROID_CORPUS.values()
]
TFIDF_VECTORIZER = TfidfVectorizer()
TFIDF_VECTORS = TFIDF_VECTORIZER.fit_transform(TOKENIZED_ANDROID_CORPUS)

##############################################################################
# Train and evaluate the models for Part 2
##############################################################################
RESULTS = []
MARGIN = 0.2
CRITERION = helpers.MaxMarginLoss(MARGIN)
MAX_EPOCHS = 50
BATCH_SIZE = 64
FILTER_WIDTH = 2
POOL_METHOD = "average"
FEATURE_DIM = 300
MODELS = [models.CNN(EMBEDDINGS, FILTER_WIDTH, POOL_METHOD, FEATURE_DIM)] # models.LSTM(...)
for model in MODELS:
    #  (use mean reciprocal rank to determine best epoch)
    OPTIMIZER = torch.optim.Adam(model.parameters(), lr=1E-3)
    result = train_utils.train_model(model, OPTIMIZER, CRITERION, DATA, \
                                    MAX_EPOCHS, BATCH_SIZE, CUDA)
    RESULTS.append(result)

##############################################################################
# Print out the results and evaluate on the test set for Part 2
##############################################################################
# Print out the results...
