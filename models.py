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
        self.embedding_layer.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_units, batch_first=True, bidirectional=False)  # Input dim is 3, output dim is 3
        self.hidden_dim = hidden_units
        self.pool_method = pool_method
        self.cuda_on = cuda
        self.bidirectional = False

    def init_hidden(self, sz):
        if self.bidirectional:
            factor = 2
        else:
            factor = 1
        if self.cuda_on:
            return (autograd.Variable(torch.zeros(factor, sz, self.hidden_dim)).cuda(),
                    autograd.Variable(torch.zeros(factor, sz, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(factor, sz, self.hidden_dim)),
                    autograd.Variable(torch.zeros(factor, sz, self.hidden_dim)))

    def forward(self, word_indices):
        embeddings = self.embedding_layer(word_indices)
        hidden = self.init_hidden(embeddings.size(0))
        lstm_out, hidden = self.lstm(embeddings.float(), hidden)
        if self.pool_method == "max":
            return hidden
        elif self.pool_method == "average":
            return torch.mean(lstm_out, 1)
        else:
            raise ValueError("Invalid self.pool_method: " + str(self.pool_method))

class CNN(nn.Module):
    def __init__(self, embeddings, filter_width, pool_method, feature_dim, dropout_p):
        super(CNN, self).__init__()

        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)

        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.weight.requires_grad = False

        self.conv2d = nn.Conv2d(1, feature_dim, (filter_width, embed_dim)).double()

        self.dropout = nn.Dropout2d(p=dropout_p)

        self.pool_method = pool_method

    def forward(self, word_indicies):
        embeddings = self.embedding_layer(word_indicies)
        convolved = self.conv2d(embeddings.unsqueeze(1).double())
        activation = F.tanh(convolved.squeeze(3))
        activation = self.dropout(activation)
        if self.pool_method == "max":
            return torch.max(activation, 2)
        elif self.pool_method == "average":
            return torch.mean(activation, 2)
        else:
            raise ValueError("Invalid self.pool_method: " + str(self.pool_method))

class BinaryClassifier(nn.Module):
    def __init__(self, question_encoding_size, num_hidden_units):
        super(BinaryClassifier, self).__init__()

        self.fc1 = nn.Linear(question_encoding_size, num_hidden_units)
        self.o = nn.Linear(num_hidden_units, 2)

    def forward(self, question_encoding):
        z = self.fc1(question_encoding)
        a = F.relu(z)
        out = self.o(a)
        return F.log_softmax(out)