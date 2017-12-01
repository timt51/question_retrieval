import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, embeddings):
        super(LSTM, self).__init__()

        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size,
                                            embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)

    def forward():
        pass

class CNN(nn.Module):
    def __init__(self, embeddings, filter_width, pool_method, feature_dim):
        super(CNN, self).__init__()

        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size,
                                            embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)

        self.conv2d = nn.Conv2d(1, feature_dim, (filter_width, embed_dim)).double()

        self.pool_method = pool_method

    def forward(self, word_indicies):
        embeddings = self.embedding_layer(word_indicies)
        convolved = self.conv2d(embeddings.unsqueeze(1).double())
        activation = F.tanh(convolved.squeeze(3))
        if self.pool_method == "max":
            return torch.max(activation, 2)
        elif self.pool_method == "average":
            return torch.mean(activation, 2)
        else:
            raise ValueError("Invalid self.pool_method: " + str(self.pool_method))
