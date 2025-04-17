from collections import OrderedDict

from torchaudio.models import Conformer
from torchaudio.transforms import TimeMasking
import torch
import torch.nn as nn
import torch.nn.functional as F

class Wav2Vec(nn.Module):
    def __init__(self, input_dim, embed_dim, ffn_dim, depthwise_conv_kernel_size, num_heads, num_conformers, mask_ratio=0.2, distraction_ratio=0.2):
        super(Wav2Vec, self).__init__()
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
        self.target_encoder = nn.Linear(embed_dim, embed_dim)
        self.masking_prob = mask_ratio
        self.distraction_prob = distraction_ratio

    def _apply_conv(self, x):
        x = x.transpose(1, 2)
        x = self.subsample(x)
        x = x.transpose(1, 2)
        return x

    def _apply_mask(self, tensor):
        bs, T, a = tensor.size()
        mask = torch.ones(T, dtype=torch.bool)
        num_masked_rows = int(self.masking_prob * T)
        masked_indices = torch.randperm(T)[:num_masked_rows]
        mask[masked_indices] = 0
        masked_tensor = tensor.clone()
        masked_tensor[:, ~mask, :] = 0
        return masked_tensor, masked_indices

    def _get_distract_vectors(self, tensor, target_indices):
        bs, T, a = tensor.size()
        mask = torch.ones(T, dtype=torch.bool)
        num_masked_rows = int(self.distraction_prob * T)
        masked_indices = torch.randperm(T)[:num_masked_rows]
        mask[masked_indices] = 0
        mask[target_indices] = 1
        masked_tensor = tensor.clone()
        #masked_tensor[:, ~mask, :] = 0
        return masked_tensor[:, ~mask, :]

    def forward(self, x, key_padding_masks):
        # Subsampling
        x = x[:, :torch.max(key_padding_masks), :]
        x = self._apply_conv(x) # -> subsampling with conv layer

        # Getting context vectors
        c, masked_indices = self._apply_mask(x) # -> Masking over time stamps
        c = self.linear_layer(c) # -> first layer of a context network
        c, _ = self.conformer(c, key_padding_masks) # -> conformers in a context network
        c = c[:, masked_indices, :] # -> getting vectors on initially masked positions

        # Getting target vectors
        x = self.target_encoder(x) # -> target encoding

        y_t = x[:, masked_indices, :]
        y_distraction = self._get_distract_vectors(x, masked_indices)

        return c, y_t, y_distraction

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.k = temperature

    def forward(self, c, y_t, y_distraction):
        # Positive logits
        q_t = F.cosine_similarity(y_t, c, dim=-1) / self.k  # (bs, t)
        q_t = torch.exp(q_t)

        # Negative logits
        c_expanded = c.unsqueeze(2)
        y_expanded = y_distraction.unsqueeze(1)
        q_neg = torch.exp(F.cosine_similarity(c_expanded, y_expanded, dim=-1) / self.k)  # (bs, t, k)

        # Denominator: positive + negatives
        denom = q_neg.sum(dim=-1) + q_t

        # Final loss (mean over all time steps and batch)
        loss = -torch.log(q_t / denom).mean()
        return loss

