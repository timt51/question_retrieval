from collections import namedtuple

##############################################################################
# Settings
##############################################################################
CUDA = False

##############################################################################
# Load the dataset
##############################################################################
# CORPUS = dict: id -> namedtuple(title, body)
# TRAIN_DATA = dict: (id, id_of_similar) -> set: negative_ids
# DEV_DATA = dict: id -> namedtuple(ids_of_similar, ids_of_candidates)
# TEST_DATA = dict: id -> namedtuple(ids_of_similar, ids_of_candidates)
# EMBEDDINGS, WORD_TO_INDEX =
#   np_array: [vocab_size, embedding_size], dict: word -> index

##############################################################################
# Train and evaluate the models
##############################################################################
Result = namedtuple("Result", \
        "model lr other_hyperparameters map mrr pat1 pat5")
# RESULTS = []
# OPTIMIZER = ...
# CRITERION = helper.MaxMarginLoss(margin)
# MAX_EPOCHS = ...
# BATCH_SIZE = ...
# MODELS = [models.LSTM(...), models.CNN(...)]
# for model in MODELS:
#   (use mean reciprocal rank to determine best epoch)
#   result =
#   train_utils.train_model(model, OPTIMIZER, CRITERION, TRAIN_DATA, \
#                           DEV_DATA, MAX_EPOCHS, BATCH_SIZE, CUDA)
#   RESULTS.append(result)

##############################################################################
# Print out the results and evaluate on the test set
##############################################################################
# Print out the results...
