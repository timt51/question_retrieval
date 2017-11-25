import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embeddings):
        super(LSTM, self).__init__()

        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)

    def forward():
        pass

class CNN(nn.Module):
    def __init__(self, embeddings):
        super(CNN, self).__init__()

        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)

    def forward():
        pass
