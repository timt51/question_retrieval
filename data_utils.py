import os
import subprocess

ASK_UBUNTU_GIT_REPO = "https://github.com/taolei87/askubuntu.git"
ASK_UBUNTU_GIT_REPO_PATH = "./askubuntu"

def download_ask_ubuntu_dataset():
    """
    Download the dataset if it does not already exists
    """
    if not os.path.isdir(ASK_UBUNTU_GIT_REPO_PATH):
        try:
            subprocess.call(["git", "clone", ASK_UBUNTU_GIT_REPO])
        except subprocess.SubprocessError:
            print("Error: May have failed to download the dataset")

# CORPUS = dict: id -> namedtuple(title, body)
# TRAIN_DATA = dict: (id, id_of_similar) -> set: negative_ids
# DEV_DATA = dict: id -> namedtuple(ids_of_similar, ids_of_candidates)
# TEST_DATA = dict: id -> namedtuple(ids_of_similar, ids_of_candidates)
# EMBEDDINGS, WORD_TO_INDEX =
#   np_array: [vocab_size, embedding_size], dict: word -> index


# (Note that there also exists a list of held out ids)
