import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from networks.model.ops.attention import MultiHeadAttention, MultiHeadCustomAttention


class justatt(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, similiarity='scaled_dot'):
        super(justatt, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn = MultiHeadAttention(d_model, nhead)
        # self.self_attn = MultiHeadCustomAttention(d_model, nhead, similiarity)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar): # Transformer中的EncoderLayer
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src = torch.unsqueeze(src, 1) # B, C -> B, T, C
        tar = torch.unsqueeze(tar, 1) # B, C -> B, T, C
        src2 = self.self_attn(tar, src, src)[0]
        src = torch.squeeze(src2, 1) # B, T, C -> B, C
        return src


class withoutatt(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, similiarity='scaled_dot'):
        super(withoutatt, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn = MultiHeadAttention(d_model, nhead)
        # self.self_attn = MultiHeadCustomAttention(d_model, nhead, similiarity)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar): # Transformer中的EncoderLayer
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src = torch.unsqueeze(src, 1) # B, C -> B, T, C
        tar = torch.unsqueeze(tar, 1) # B, C -> B, T, C
        src2 = src
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = torch.squeeze(src, 1) # B, T, C -> B, C
        return src


class attbase(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, similiarity='scaled_dot'):
        super(attbase, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.self_attn = MultiHeadAttention(d_model, nhead)
        self.self_attn = MultiHeadCustomAttention(d_model, nhead, similiarity)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar): # Transformer中的EncoderLayer
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src = torch.unsqueeze(src, 1) # B, C -> B, T, C
        tar = torch.unsqueeze(tar, 1) # B, C -> B, T, C
        src2 = self.self_attn(tar, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = torch.squeeze(src, 1) # B, T, C -> B, C
        return src


class attnormbase(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, similiarity='scaled_dot'):
        super(attnormbase, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.self_attn = MultiHeadAttention(d_model, nhead)
        self.self_attn = MultiHeadCustomAttention(d_model, nhead, similiarity)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar): # Transformer中的EncoderLayer
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src = torch.unsqueeze(src, 1) # B, C -> B, T, C
        tar = torch.unsqueeze(tar, 1) # B, C -> B, T, C
        tar = self.norm1(tar)
        src = self.norm1(src)
        src2 = self.self_attn(tar, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = torch.squeeze(src, 1) # B, T, C -> B, C
        return src