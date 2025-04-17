from torchaudio.models import Conformer
import torch
import torch.nn as nn

class ContextVec(nn.Module):
    def __init__(self, input_dim, embed_dim, ffn_dim, depthwise_conv_kernel_size, num_heads, num_conformers, mask_ratio=0.2, distraction_ratio=0.2):
        super(ContextVec, self).__init__()
        # (bs, T, input_dim) -> (bs, T, embed_dim)

        self.subsample = nn.Conv1d(in_channels=input_dim,
                                   out_channels=embed_dim,
                                   kernel_size=depthwise_conv_kernel_size,
                                   stride=1,
                                   padding=depthwise_conv_kernel_size // 2)
        self.linear_layer = nn.Linear(in_features=embed_dim,
                                      out_features=embed_dim)
        self.conformer = Conformer(
                input_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                num_layers=num_conformers,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=0.2,)

    def _apply_conv(self, x):
        x = x.transpose(1, 2)
        x = self.subsample(x)
        x = x.transpose(1, 2)
        return x

    def forward(self, x, key_padding_masks):
        # Subsampling
        x = x[:, :torch.max(key_padding_masks), :]
        x = self._apply_conv(x) # -> subsampling with conv layer

        c = self.linear_layer(x) # -> first layer of a context network
        c, _ = self.conformer(c, key_padding_masks) # -> conformers in a context network

        return c

class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, ffn_dim, depthwise_conv_kernel_size, num_heads, num_layers):
        super().__init__()
        self.subsample = nn.Conv1d(input_dim, embed_dim, kernel_size=depthwise_conv_kernel_size, padding=depthwise_conv_kernel_size//2, stride=1)
        self.linear_layer = nn.Linear(embed_dim, embed_dim)
        self.conformer = Conformer(
            input_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=0.1
        )
        self.embed_dim = embed_dim

    def forward(self, x, lengths):
        x = x[:, :torch.max(lengths), :]
        x = x.transpose(1, 2)
        x = self.subsample(x).transpose(1, 2)
        x = self.linear_layer(x)
        x, _ = self.conformer(x, lengths)
        return x # (B, T, D)

class RNNTDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, y):
        y = self.embedding(y)  # (B, U) -> (B, U, E)
        output, _ = self.rnn(y)
        return output  # (B, U, H)

class JointNetwork(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, joint_dim, vocab_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(encoder_dim + decoder_dim, joint_dim),
            nn.ReLU(),
            nn.Linear(joint_dim, vocab_size)
        )

    def forward(self, encoder_out, decoder_out):
        # encoder_out: (B, T, D), decoder_out: (B, U, D)
        B, T, D_enc = encoder_out.size()
        B, U, D_dec = decoder_out.size()

        # Expand and merge
        enc = encoder_out.unsqueeze(2).expand(-1, T, U, -1)
        dec = decoder_out.unsqueeze(1).expand(-1, T, U, -1)

        joint = torch.cat((enc, dec), dim=-1)  # (B, T, U, D_enc + D_dec)
        out = self.fc(joint)  # (B, T, U, Vocab)
        return out

class ConformerRNNT(nn.Module):
    def __init__(self, input_dim, encoder_dim, ffn_dim, depthwise_conv_kernel_size,
                 num_heads, num_conformer_layers, vocab_size, decoder_dim, joint_dim):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim, encoder_dim, ffn_dim, depthwise_conv_kernel_size, num_heads, num_conformer_layers)
        self.decoder = RNNTDecoder(vocab_size, decoder_dim, decoder_dim)
        self.joint = JointNetwork(encoder_dim, decoder_dim, joint_dim, vocab_size)

    def forward(self, x, x_lengths, y):
        encoder_out = self.encoder(x, x_lengths)  # (B, T, D)
        decoder_out = self.decoder(y)             # (B, U, D)
        logits = self.joint(encoder_out, decoder_out)  # (B, T, U, Vocab)
        return logits

class CTCConformer(nn.Module):
    def __init__(self, conformer_encoder: ConformerEncoder, vocab_size):
        super().__init__()
        self.encoder = conformer_encoder  # Предобученный Conformer
        self.linear = nn.Linear(conformer_encoder.embed_dim, vocab_size)  # Проекция на алфавит

    def forward(self, x, input_lengths):
        # x: [batch, time, features] (например, спектрограммы)
        # input_lengths: длины аудио в батче

        # Пропускаем через энкодер
        encoder_out = self.encoder(x, input_lengths)  # [B, T, D]

        # Проекция на алфавит
        logits = self.linear(encoder_out)  # [B, T, vocab_size]

        return logits