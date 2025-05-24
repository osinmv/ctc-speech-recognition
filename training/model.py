import torch.nn as nn


class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # output_dim = 128 for ASCII characters

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch_size, seq_len, hidden_dim)
        output = self.fc(rnn_out)  # output: (batch_size, seq_len, output_dim)
        return output

SimpleRNNmodel = SimpleRNNModel(13,128,128)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # output_dim = 128 for ASCII characters
        self._log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch_size, seq_len, hidden_dim)
        output = self._log_softmax(self.fc(rnn_out))  # output: (batch_size, seq_len, output_dim)

        return output

SimpleLSTMModel =  LSTMModel(13, 128, 128)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)  # Input embedding layer

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Output projection layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self._log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)

        x = x.permute(1, 0, 2)  # Change shape to (seq_len, batch_size, hidden_dim)

        x = self.transformer_encoder(x)

        # (seq_len, batch_size, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        x = x.permute(1, 0, 2)  # Change back to (batch_size, seq_len, hidden_dim)

        output = self.fc(x)  # (batch_size, seq_len, output_dim)

        output = self._log_softmax(output)  # (batch_size, seq_len, output_dim)

        return output

SimpleTransformerModel =  TransformerModel(13, 128, 28, num_heads=16, num_layers=2)


class MediumLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,  num_layers, output_dim):
        super(MediumLSTMModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # output_dim = 128 for ASCII characters
        self._log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch_size, seq_len, hidden_dim)
        output = self._log_softmax(self.fc(rnn_out))  # output: (batch_size, seq_len, output_dim)

        return output

MyMediumLSTMModel =  MediumLSTMModel(13, 256, 4, 28)


class CharLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden
