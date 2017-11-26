import os
import subprocess
import gzip
from collections import namedtuple

import numpy as np

# (Note that there also exists a list of held out ids)

ASK_UBUNTU_GIT_REPO = "https://github.com/taolei87/askubuntu.git"
ASK_UBUNTU_GIT_REPO_PATH = "./askubuntu/"
TOKENIZED_CORPUS_PATH = ASK_UBUNTU_GIT_REPO_PATH + "text_tokenized.txt.gz"
TRAIN_DATA_PATH = ASK_UBUNTU_GIT_REPO_PATH + "train_random.txt"
DEV_DATA_PATH = ASK_UBUNTU_GIT_REPO_PATH + "dev.txt"
TEST_DATA_PATH = ASK_UBUNTU_GIT_REPO_PATH + "test.txt"
PRUNED_EMEDDINGS_PATH = ASK_UBUNTU_GIT_REPO_PATH + \
                        "vector/vectors_pruned.200.txt.gz"

CorpusEntry = namedtuple("CorpusEntry", "title body")
EvalQueryResult = namedtuple("EvalQueryResult", "similar_ids candidate_ids")

def download_ask_ubuntu_dataset():
    """
    Download the dataset if it does not already exists
    """
    if not os.path.isdir(ASK_UBUNTU_GIT_REPO_PATH):
        try:
            subprocess.call(["git", "clone", ASK_UBUNTU_GIT_REPO])
        except subprocess.SubprocessError:
            print("Error: May have failed to download the dataset")

def sequence_to_indicies(sequence, word_to_index):
    return [word_to_index.get(word, len(word_to_index)) for word in sequence]

def load_corpus(word_to_index):
    """
    Returns dict: id -> namedtuple(title, body)
    title and body are encoded as np arrays of indicies using word_to_index.
    For a token T not in word_to_index, word_to_index[T] = len(word_to_index).
    """
    corpus = {}
    with gzip.open(TOKENIZED_CORPUS_PATH, 'rt', encoding="utf8") as file:
        for line in file:
            entry_id, title, body = line.split("\t")
            entry_id = int(entry_id)
            title = sequence_to_indicies(title.split(" "), word_to_index)
            body = sequence_to_indicies(body.split(" "), word_to_index)
            corpus[entry_id] = CorpusEntry(np.array(title), np.array(body))
    return corpus

def load_train_data():
    """
    Returns dict: (entry_id, similar_id) -> set: negative_ids
    """
    train_data = {}
    with open(TRAIN_DATA_PATH, "rt") as file:
        for line in file:
            entry_id, similar_ids, negative_ids = line.split("\t")
            entry_id = int(entry_id)
            negative_ids = [int(x) for x in negative_ids.split(" ")]
            for similar_id in map(int, similar_ids.split(" ")):
                train_data[(entry_id, similar_id)] = negative_ids
    return train_data

def process_eval_data(data_path):
    """
    Returns dict: id -> namedtuple(ids_of_similar, ids_of_candidates)
    """
    data = {}
    with open(data_path, "rt") as file:
        for line in file:
            entry_id, similar_ids, candidate_ids, _ = line.split("\t")
            if similar_ids == "":
                continue
            entry_id = int(entry_id)
            similar_ids = [int(x) for x in similar_ids.split(" ")]
            candidate_ids = [int(x) for x in candidate_ids.split(" ")]
            data[entry_id] = EvalQueryResult(similar_ids, candidate_ids)
    return data

def load_eval_data():
    """
    Returns dict: id -> namedtuple(ids_of_similar, ids_of_candidates)
            for dev and test sets
    """
    dev_data = process_eval_data(DEV_DATA_PATH)
    test_data = process_eval_data(TEST_DATA_PATH)
    return dev_data, test_data

def load_embeddings():
    """
    Returns np_array: [vocab_size, embedding_size], dict: word -> index
    """
    embeddings = []
    word_to_index = {}
    with gzip.open(PRUNED_EMEDDINGS_PATH, 'rt', encoding="utf8") as file:
        for index, line in enumerate(file):
            line = line.rstrip().split(" ")
            word_to_index[line[0]] = index
            embeddings.append(np.array([float(x) for x in line[1:]]))
    return np.array(embeddings), word_to_index
