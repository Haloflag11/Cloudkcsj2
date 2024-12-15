import concurrent.futures
import threading
from layers.LGMixer_Modified import LGMixer
import torch.nn as nn
import torch.nn.functional as F
import torch
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class AttentionMixer(nn.Module):
    def __init__(self, num_attn_types=2):
        super(AttentionMixer, self).__init__()
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))  # 初始化为 0.0 为了得到 sigmoid(0) = 0.5

    def forward(self, *attns):
        if len(attns) != 2:
            raise ValueError("AttentionMixer requires exactly two attention maps.")
        alpha = torch.sigmoid(self.raw_alpha)
        mixed_attention=(alpha*attns[0]+(1-alpha)*attns[1])
        return mixed_attention

class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="swish"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_r = attention_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        #self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.silu
        self.alpha = nn.Parameter(torch.tensor(0.5)) 
        self.mixer1=LGMixer(d_model, d_ff)
        self.mixer2= AttentionMixer()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
            nn.Dropout(0.1)
        ) if d_model and d_ff else Non
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        forward_attn = self.attention(x)
        reversed_attn=self.attention_r(x.flip(dims=[1])).flip(dims=[1])
        attns=[forward_attn,reversed_attn]
        #Gated TD Encoder
        new_x=forward_attn+reversed_attn
        attn = 1
        x = x+new_x
        y = x = self.norm1(x)
        y=self.mixer1(y)
        e_out=self.norm2(self.mixer2(x,y))
        
        # if self.ffn is not None:
        #     _x=self.ffn(m_out)
            
        return e_out, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x=self.norm(x)

        return x, attns

