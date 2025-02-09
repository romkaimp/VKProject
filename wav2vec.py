from collections import OrderedDict

from torchaudio.models import Conformer
from torchaudio.transforms import TimeMasking
import torch
import torch.nn as nn
import torch.nn.functional as F

class Wav2Vec(nn.Module):
    def __init__(self, input_dim, embed_dim, ffn_dim, depthwise_conv_kernel_size, num_heads, N, mask_ratio=0.2, distraction_ratio=0.2):
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
                num_layers=N,
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
        x = x[:, :torch.max(key_padding_masks), :]
        x = self._apply_conv(x) # -> subsampling with conv layer
        y, masked_indices = self._apply_mask(x) # -> Masking over time stamps
        y = self.linear_layer(y) # -> first layer of a context network
        #print(y.shape, key_padding_masks.shape)
        y, _ = self.conformer(y, key_padding_masks) # -> conformers in a context network
        x = self.target_encoder(x) # -> target encoding

        y = y[:, masked_indices, :]
        x_t = x[:, masked_indices, :]
        x_distraction = self._get_distract_vectors(x, masked_indices)

        return y, x_t, x_distraction

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.k = temperature

    def forward(self, c, y_t, y_distraction):
        super(ContrastiveLoss, self).__init__()
        # c ~ (bs, t, a)
        q_t = F.cosine_similarity(y_t, c, dim=-1) # (bs, t)
        q_t = torch.exp(q_t/self.k)

        c_expanded = c.unsqueeze(2) # (bs, t, a) -> (bs, t, 1, a)
        y_expanded = y_distraction.unsqueeze(1) # (bs, k, a) -> (bs, 1, k, a)

        q_distraction = F.cosine_similarity(y_expanded, c_expanded, dim=-1) # (bs, t, k)
        q_distraction = torch.exp(q_distraction/self.k)
        q_distraction = q_distraction.sum(dim=2) + q_t # (bs, t)

        loss = -torch.sum(torch.log(torch.div(q_t, q_distraction)), dim=(0, 1))
        return loss
