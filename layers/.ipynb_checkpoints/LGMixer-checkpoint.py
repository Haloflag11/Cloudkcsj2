import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class LGMixer(nn.Module):
    def __init__(self, d_model, d_ff, num_attn_types=2):
        super(LGMixer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)

    def forward(self, attn):


        # Conv1 后进行转置（batch_size, d_model, seq_len）
        alpha = F.gelu(self.conv1(attn.transpose(-1, 1)).transpose(-1, 1))
        beta = self.conv2(attn.transpose(-1, 1)).transpose(-1, 1)  # Conv2 后进行转置
        assert alpha.shape == beta.shape, f"Shape mismatch: alpha {alpha.shape}, beta {beta.shape}"

        return alpha * beta