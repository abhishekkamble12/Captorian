import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim=2048, decoder_dim=512, attention_dim=256):
        super().__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        hidden = decoder_hidden.unsqueeze(1)
        energy = torch.tanh(self.W_encoder(encoder_out) + self.W_decoder(hidden))
        weights = F.softmax(self.V(energy), dim=1)
        context = torch.sum(encoder_out * weights, dim=1)
        return context, weights
