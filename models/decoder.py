import torch
import torch.nn as nn

from .attention import BahdanauAttention


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, encoder_dim=2048, decoder_dim=512, attention_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def init_hidden_state(self, encoder_out):
        mean_features = encoder_out.mean(dim=1)
        return torch.tanh(self.init_h(mean_features)), torch.tanh(self.init_c(mean_features))

    def forward(self, encoder_out, captions):
        embeddings = self.dropout(self.embedding(captions))
        hidden, cell = self.init_hidden_state(encoder_out)
        outputs = []
        for time_step in range(captions.size(1)):
            context, _ = self.attention(encoder_out, hidden)
            inputs = torch.cat([embeddings[:, time_step, :], context], dim=1)
            hidden, cell = self.lstm_cell(inputs, (hidden, cell))
            outputs.append(self.fc(self.dropout(hidden)).unsqueeze(1))
        return torch.cat(outputs, dim=1)
