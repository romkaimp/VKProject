import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Модель трансформера
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(TransformerLanguageModel, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers
        )
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, tgt_mask=None, lengths=None, tgt_key_padding_mask=None):
        tgt = tgt[:, :torch.max(lengths).item()]

        if tgt.size(0) == 0:
            return torch.zeros(0, tgt.size(1), self.linear.out_features, device=tgt.device)

        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = tgt.transpose(0, 1)
        tgt = self.pos_encoder(tgt)
        print(tgt.shape)

        memory = torch.zeros((tgt.size(1), tgt.size(0), self.d_model), device=tgt.device)  # Пустой источник
        memory = self.pos_encoder(memory)

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.linear(output.transpose(0, 1))

        return output
