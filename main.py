from collections import namedtuple
import itertools

import torch

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
CORPUS = data_utils.load_corpus(WORD_TO_INDEX)
TRAIN_DATA = data_utils.load_train_data()
DEV_DATA, TEST_DATA = data_utils.load_eval_data()
DATA = Data(CORPUS, TRAIN_DATA, DEV_DATA, TEST_DATA,\
            EMBEDDINGS, WORD_TO_INDEX)

##############################################################################
# Train and evaluate the models for Part 1
##############################################################################
RESULTS = []
MARGINS = [0.2, 0.4, 0.5, 0.6, 0.8]
MAX_EPOCHS = 50
BATCH_SIZE = 32
FILTER_WIDTHS = [3]
POOL_METHOD = "average"
FEATURE_DIMS = [600, 667]
DROPOUT_PS = [0.1, 0.2, 0.3]
NUM_HIDDEN_UNITS = [150, 200, 240, 280, 350]
LEARNING_RATES = [3E-4, 1E-3]
MODELS = []
##############################################################################
# LSTM_HYPERPARAMETERS = itertools.product(MARGINS, NUM_HIDDEN_UNITS, LEARNING_RATES)
# for margin, num_hidden_units, learning_rate in LSTM_HYPERPARAMETERS:
#     model = models.LSTM(EMBEDDINGS, num_hidden_units, POOL_METHOD, CUDA)
#     criterion = helpers.MaxMarginLoss(margin)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     model, mrr = train_utils.train_model(model, optimizer, criterion, DATA, \
#                                     MAX_EPOCHS, BATCH_SIZE, CUDA)
#     torch.save(model.state_dict(), "./models/lstm_" +
#                                     str(margin) + "_" +
#                                     str(num_hidden_units) + "_" +
#                                     str(learning_rate))
#     MODELS.append((mrr, margin, num_hidden_units, learning_rate))
##############################################################################
CNN_HYPERPARAMETERS = itertools.product(MARGINS, FILTER_WIDTHS, FEATURE_DIMS, 
                                        DROPOUT_PS, LEARNING_RATES)
for margin, filter_width, feature_dim, dropout_p, learning_rate in CNN_HYPERPARAMETERS:
    model = models.CNN(EMBEDDINGS, filter_width, POOL_METHOD, feature_dim, dropout_p)
    criterion = helpers.MaxMarginLoss(margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, mrr = train_utils.train_model(model, optimizer, criterion, DATA, \
                                    MAX_EPOCHS, BATCH_SIZE, CUDA)
    torch.save(model.state_dict(), "./models/cnn_" +
                                    str(margin) + "_" +
                                    str(filter_width) + "_" +
                                    str(feature_dim) + "_" +
                                    str(dropout_p) + "_" +
                                    str(learning_rate))
    print("should have saved", "./models/cnn_" +
                                    str(margin) + "_" +
                                    str(filter_width) + "_" +
                                    str(feature_dim) + "_" +
                                    str(dropout_p) + "_" +
                                    str(learning_rate))
    MODELS.append((mrr, margin, filter_width, feature_dim, dropout_p, learning_rate))

##############################################################################
# Print out the results and evaluate on the test set for Part 1
##############################################################################
# Print out the results...
MODELS = sorted(MODELS, key=lambda x: x[0], reverse=True)
for model in MODELS:
    print(model)
