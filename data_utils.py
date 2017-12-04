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

ANDROID_GIT_REPO = "https://github.com/jiangfeng1124/Android"
ANDROID_GIT_REPO_PATH = "./Android/"
TOKENIZED_ANDROID_CORPUS_PATH = ANDROID_GIT_REPO_PATH + "corpus.tsv.gz"
ANDROID_POS_DEV_DATA_PATH = ANDROID_GIT_REPO_PATH + "dev.pos.txt"
ANDROID_NEG_DEV_DATA_PATH = ANDROID_GIT_REPO_PATH + "dev.neg.txt"
ANDROID_POS_TEST_DATA_PATH = ANDROID_GIT_REPO_PATH + "test.pos.txt"
ANDROID_NEG_TEST_DATA_PATH = ANDROID_GIT_REPO_PATH + "test.neg.txt"

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
    embeddings.append(np.array([0 for _ in range(len(embeddings[0]))]))
    return np.array(embeddings), word_to_index

def download_android_dataset():
    """
    Download the dataset if it does not already exists
    """
    if not os.path.isdir(ANDROID_GIT_REPO_PATH):
        try:
            subprocess.call(["git", "clone", ANDROID_GIT_REPO])
        except subprocess.SubprocessError:
            print("Error: May have failed to download the dataset")

def load_android_corpus(word_to_index):
    """
    Returns dict: id -> namedtuple(title, body)
    title and body are encoded as np arrays of indicies using word_to_index.
    For a token T not in word_to_index, word_to_index[T] = len(word_to_index).
    """
    corpus = {}
    with gzip.open(TOKENIZED_ANDROID_CORPUS_PATH, 'rt', encoding="utf8") as file:
        for line in file:
            entry_id, title, body = line.split("\t")
            entry_id = int(entry_id)
            title = sequence_to_indicies(title.split(" "), word_to_index)
            body = sequence_to_indicies(body.split(" "), word_to_index)
            corpus[entry_id] = CorpusEntry(np.array(title), np.array(body))
    return corpus

def process_android_eval_data(positive_path, negative_path):
    """
    Returns dict: id -> namedtuple(ids_of_similar, ids_of_candidates)
    """
    positives = {}
    with open(positive_path, "rt") as file:
        for line in file:
            entry_id, positive = line.split(" ")
            entry_id, positive = int(entry_id), int(positive)
            if entry_id not in positives:
                positives[entry_id] = [positive]
            else:
                positives[entry_id].append(positive)
    negatives = {}
    with open(negative_path, "rt") as file:
        for line in file:
            entry_id, negative = line.split(" ")
            entry_id, negative = int(entry_id), int(negative)
            if entry_id not in negatives:
                negatives[entry_id] = [negative]
            else:
                negatives[entry_id].append(negative)
    data = {}
    for entry_id in positives:
        similar_ids = positives[entry_id]
        candidate_ids = positives[entry_id] + negatives[entry_id]
        data[entry_id] = EvalQueryResult(similar_ids, candidate_ids)
    return data

def load_android_eval_data():
    """
    Returns dict: id -> namedtuple(ids_of_similar, ids_of_candidates)
            for dev and test sets
    """
    dev_data = process_android_eval_data(ANDROID_POS_DEV_DATA_PATH, ANDROID_NEG_DEV_DATA_PATH)
    test_data = process_android_eval_data(ANDROID_POS_TEST_DATA_PATH, ANDROID_NEG_TEST_DATA_PATH)
    return dev_data, test_data

def load_tokenized_android_corpus(word_to_index):
    """
    Returns dict: id -> namedtuple(title, body)
    title and body are encoded as np arrays of indicies using word_to_index.
    For a token T not in word_to_index, word_to_index[T] = len(word_to_index).
    """
    corpus = {}
    with gzip.open(TOKENIZED_ANDROID_CORPUS_PATH, 'rt', encoding="utf8") as file:
        for line in file:
            entry_id, title, body = line.split("\t")
            entry_id = int(entry_id)
            corpus[entry_id] = CorpusEntry(title.join(" "), body.join(" "))
    return corpus
