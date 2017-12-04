import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class LSTM(nn.Module):
    def __init__(self, embeddings, hidden_units, pool_method, cuda):
        super(LSTM, self).__init__()

        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.lstm = nn.LSTM(embed_dim, hidden_units, batch_first=True)  # Input dim is 3, output dim is 3
        self.hidden_dim = hidden_units
        self.pool_method = pool_method
        self.cudaOn = cuda

    def init_hidden(self, sz):
        if self.cudaOn:
            return (autograd.Variable(torch.zeros(1, sz, self.hidden_dim)).cuda(),
                    autograd.Variable(torch.zeros(1, sz, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1, sz, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, sz, self.hidden_dim)))

    def forward(self, word_indices):
        # alternatively, we can do the entire sequence all at once.
        # the first value returned by LSTM is all of the hidden states throughout
        # the sequence. the second is just the most recent hidden state
        # Add the extra 2nd dimension
        embeddings = self.embedding_layer(word_indices)
        hidden = self.init_hidden(embeddings.size(0))
        lstm_out, hidden = self.lstm(embeddings.float(), hidden)
        if self.pool_method == "max":
            return hidden
        elif self.pool_method == "average":
            return torch.mean(lstm_out, 1)
        else:
            raise ValueError("Invalid self.pool_method: " + str(self.pool_method))

        # dd9*0c72L884
        # TODO: implement me!!

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
