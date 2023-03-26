import torch
from torch import nn, Tensor
from torch._C import device
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.pooling import MaxPool2d
import torch.utils.model_zoo as model_zoo

import torchvision.transforms as transforms
import numpy as np
import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 501):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class AttBlock(nn.Module):
    def __init__(self, d_model, nhead: int = 4):
        super(AttBlock, self).__init__()
        self.multi_att = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)


    def forward(self, X, Y, attn_mask, key_padding_mask):
        '''
        X:      query (L, N, E)
        Y:      key value (S, N, E)
        atten_mask: (L, S)
        key_padding_mask: `(N, S)`
        output: (L, N, E)
        '''
        # H = self.norm1(self.multi_att(X, Y, Y, attn_mask = mask)[0] + X)
        # return self.norm2(torch.relu(self.linear(H)) + H)

        ## Tr_I wrong
        # H = self.norm1(self.multi_att(X, Y, Y, attn_mask = mask)[0]) + X
        # return self.norm2(torch.relu(self.linear(H))) + H

        ## Tr_I_fix
        H = torch.relu(self.multi_att(X, Y, Y, key_padding_mask = key_padding_mask, attn_mask = attn_mask)[0]) + X
        return torch.relu(self.linear(self.norm1(H))) + H


class SMT_state_encoder(nn.Module):
    def __init__(self, d_model, nhead: int = 4):
        super(SMT_state_encoder, self).__init__()
        self.encoder = AttBlock(d_model, nhead)  ##0处是值，1是权重
        self.decoder = AttBlock(d_model, nhead)
        self.pos_encoder1 = PositionalEncoding(d_model)
        self.pos_encoder2 = PositionalEncoding(d_model)

    def forward(self, o, M, flag, key_padding_mask):

        c_mask  = self.casual_mask(M)
        M  = self.encoder(M, M, c_mask, key_padding_mask)
        if flag == 1: ## training
            attn_mask = self.sequence_length_mask(M, 32)   ## T * T 
        else:         ## inference
            attn_mask = self.infer_mask(o, M, 32)
        return self.decoder(o, M, attn_mask, key_padding_mask)

    def casual_mask(self, seq):
        seq_len, batch_size, _ = seq.size()
        mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device = 'cuda'),
                        diagonal=1)  ## 1 为mask
        # mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
        return mask.to(bool)

    def infer_mask(self, o, M, length):
        '''
        o_len * M_len
        '''
        o_len, batch_size, _ = o.size()
        M_len, batch_size, _ = M.size()

        mask = torch.ones((o_len, M_len), dtype=torch.uint8, device = 'cuda')
        mask[:, max(M_len - length, 0) : M_len] = 0 ## 1是mask
        return mask.to(bool)

    def sequence_length_mask(self, seq, length):
        seq_len, batch_size, _ = seq.size()
        casual_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device = 'cuda'),
                        diagonal=1)  ## 1 为mask casual_mask

        len_mask = 1 - torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device = 'cuda'),
                        diagonal= -(length - 1))    # 输入矩阵保留主对角线与主对角线以上与主对角线下方h行对角线的元素
        mask = casual_mask + len_mask
        return mask.to(bool).cuda()
## pytorch N C HW
## tensorflow NHW C
if __name__ == '__main__':
    # encoder = Encoder(feature_dim = 252)
    # rgb = Variable(torch.randn(1, 3, 180, 320))
    # depth = Variable(torch.randn(1, 1, 180, 320))
    o_obs = Variable(torch.randn(1, 1, 4))
    M_obs = Variable(torch.randn(33, 1, 4))

    # print(encoder(rgb,depth,task_obs).size())
    # print(x)

    encoder = SMT_state_encoder(512, 4)
    print(encoder.infer_mask(o_obs, M_obs, 32))





