from collections import namedtuple

import torch

import data_utils
import models
import train_utils
import helpers

##############################################################################
# Settings
##############################################################################
CUDA = False

##############################################################################
# Load the dataset
##############################################################################
Data = namedtuple("Data", \
        "corpus train dev test embeddings word_to_index")
data_utils.download_ask_ubuntu_dataset()
EMBEDDINGS, WORD_TO_INDEX = data_utils.load_embeddings()
CORPUS = data_utils.load_corpus(WORD_TO_INDEX)
TRAIN_DATA = data_utils.load_train_data()
DEV_DATA, TEST_DATA = data_utils.load_eval_data()
DATA = Data(CORPUS, TRAIN_DATA, DEV_DATA, TEST_DATA,\
            EMBEDDINGS, WORD_TO_INDEX)

##############################################################################
# Train and evaluate the models
##############################################################################
RESULTS = []
MARGIN = 1
CRITERION = helpers.MaxMarginLoss(MARGIN)
MAX_EPOCHS = 50
BATCH_SIZE = 16
FILTER_WIDTH = 3
POOL_METHOD = "average"
MODELS = [models.CNN(EMBEDDINGS, FILTER_WIDTH, POOL_METHOD)] # models.LSTM(...)
for model in MODELS:
    #  (use mean reciprocal rank to determine best epoch)
    OPTIMIZER = torch.optim.Adam(model.parameters(), lr=1E-3)
    result = train_utils.train_model(model, OPTIMIZER, CRITERION, DATA, \
                                    MAX_EPOCHS, BATCH_SIZE, CUDA)
    RESULTS.append(result)

##############################################################################
# Print out the results and evaluate on the test set
##############################################################################
# Print out the results...
