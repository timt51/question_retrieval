from collections import namedtuple

import data_utils
import models
import train_utils

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
# OPTIMIZER = ...
# CRITERION = helper.MaxMarginLoss(margin)
# MAX_EPOCHS = ...
# BATCH_SIZE = ...
# MODELS = [models.LSTM(...), models.CNN(...)]
# for model in MODELS:
#   (use mean reciprocal rank to determine best epoch)
#   result =
#   train_utils.train_model(model, OPTIMIZER, CRITERION, DATA, \
#                           MAX_EPOCHS, BATCH_SIZE, CUDA)
#   RESULTS.append(result)

##############################################################################
# Print out the results and evaluate on the test set
##############################################################################
# Print out the results...
