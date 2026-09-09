import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from models.layers import *

class _ResCNNBlock(nn.Module):
    def __init__(self, ni, nf, kss=[7, 5, 3]):
        super(_ResCNNBlock, self).__init__()
        self.convblock1 = Conv1d_new_padding(ni, nf, kss[0])
        self.convblock2 = Conv1d_new_padding(nf, nf, kss[1])
        self.convblock3 = Conv1d_new_padding(nf, nf, kss[2])
        self.bn = nn.BatchNorm1d(num_features=nf)

        # expand channels for the sum if necessary
        self.shortcut = ConvBlock(ni, nf, 1)
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.bn(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x


class ResCNN(nn.Module):
    def __init__(self, c_in, c_out):
        super(ResCNN, self).__init__()
        nf = 64
        self.block1 = _ResCNNBlock(c_in, nf, kss=[7, 5, 3])
        self.block2 = ConvBlock(nf, nf * 2, 3)
        self.act2 = nn.LeakyReLU(negative_slope = .2)
        self.block3 = ConvBlock(nf * 2, nf * 4, 3)
        self.act3 = nn.PReLU()
        self.block4 = ConvBlock(nf * 4, nf * 2, 3)
        self.act4 = nn.ELU(alpha = .3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.lin = nn.Linear(nf * 2, c_out)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.act2(x)
        x = self.block3(x)
        x = self.act3(x)
        x = self.block4(x)
        x = self.act4(x)
        embedding_output = self.squeeze(self.gap(x))
        cls_output = self.lin(embedding_output)
        return embedding_output, cls_output

if __name__ == '__main__':
    model = ResCNN(2, 16).to('cuda:0')
    print(model)
    print('-' * 100)
    x = torch.rand((5, 2, 6000), dtype=torch.float32).to('cuda:0')
    y = model(x)
    if isinstance(y, tuple):
        _, y = y
    print(y.shape)
