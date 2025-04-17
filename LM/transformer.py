import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Модель трансформера
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(TransformerLanguageModel, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=2)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # <- ВАЖНО: чтобы [B, T, D] формат был
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, tgt_mask=None, lengths=None, tgt_key_padding_mask=None):
        tgt_embed = self.embedding(tgt) * math.sqrt(self.d_model)  # [B, T, D]
        tgt_embed = self.pos_encoder(tgt_embed.transpose(0, 1)).transpose(0, 1)  # сохраняем [B, T, D]

        B, T, D = tgt_embed.shape

        # Создаём пустой источник памяти (memory), например, нули
        memory = torch.zeros(B, 1, D, device=tgt.device)

        # Приведение масок к типу bool
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.bool()
        if tgt_mask is not None:
            tgt_mask = tgt_mask.bool()

        out = self.transformer_decoder(
            tgt_embed,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.linear(out)
