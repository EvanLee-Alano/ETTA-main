import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from models.layers import *

# Cell

class Add(nn.Module):
    def forward(self, x, y):
        return x.add(y)
    def __repr__(self):
        return f'{self.__class__.__name__}'

def noop(x=None, *args, **kwargs):
    "Do nothing"
    return x

class ConvBlock(nn.Module):
    "Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."
    def __init__(self, ni, nf, kernel_size=None, stride=1, act=None, pad_zero=True):
        super(ConvBlock, self).__init__()
        kernel_size = kernel_size
        self.layer_list = []

        self.conv = Conv1d_new_padding(ni, nf, ks=kernel_size, stride=stride, pad_zero=pad_zero)
        self.bn = nn.BatchNorm1d(num_features=nf)
        self.layer_list += [self.conv, self.bn]
        if act is not None: self.layer_list.append(act)

        self.net = nn.Sequential(*self.layer_list)

    def forward(self, x):
        x = self.net(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=39, bottleneck=True, pad_zero=True):
        super(InceptionModule, self).__init__()

        bottleneck = bottleneck if ni > 1 else False  ## first layer:False
        self.bottleneck = Conv1d_new_padding(ni, nf, 1, bias=False, pad_zero=pad_zero) if bottleneck else noop
        self.convs = Conv1d_new_padding(nf if bottleneck else ni, nf * (OUT_NUM), ks, bias=False, pad_zero=pad_zero)

        self.bn = nn.BatchNorm1d(nf * OUT_NUM)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.convs(x)
        return self.act(self.bn(x))


class InceptionBlock(nn.Module):
    def __init__(self, ni, nf=47, depth=4, ks=39, pad_zero=True):
        super(InceptionBlock, self).__init__()
        self.depth = depth
        self.inception = nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModule(ni if d == 0 else nf * OUT_NUM, nf, ks=ks, pad_zero=pad_zero))

    def forward(self, x):
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
        return x


class DCN(nn.Module):
    def __init__(self, c_in, c_out, nf=47, depth=4, kernel=39, pad_zero=True):
        super(DCN, self).__init__()
        self.inceptionblock = InceptionBlock(c_in, nf, depth=depth, ks=kernel, pad_zero=pad_zero)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(nf * OUT_NUM, c_out)

    def forward(self, x):
        embedding_output = self.inceptionblock(x)
        cls_output = self.fc(self.gap(embedding_output))
        return embedding_output, cls_output

class SCNN(nn.Module):
    def __init__(self, c_in, c_out, nf=47, depth=4, kernel=39, adaptive_size=50, pad_zero=False):
        super(SCNN, self).__init__()
        self.block = InceptionBlock(c_in, nf, depth=depth, ks=kernel, pad_zero=pad_zero)
        self.head_nf = nf * OUT_NUM
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(adaptive_size),
                                  ConvBlock(self.head_nf, c_out, 1, act=None),
                                  GAP1d(1))

    def forward(self, x):
        embedding_output = self.block(x)
        cls_output = self.head(embedding_output)
        return embedding_output, cls_output


if __name__ == '__main__':
    model = DCN(2, 16).to('cuda:0')
    print(model)
    print('-' * 100)
    x = torch.rand((5, 2, 6000), dtype=torch.float32).to('cuda:0')
    y = model(x)
    print(y[0].shape)
    print(y[1].shape)
