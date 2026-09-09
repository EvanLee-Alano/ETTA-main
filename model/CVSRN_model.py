#from utils.complexcnn import *
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x_real = x[:, 0:x.shape[1] // 2, :]
        x_img = x[:, x.shape[1] // 2: x.shape[1], :]
        real = self.conv_re(x_real) - self.conv_im(x_img)
        imaginary = self.conv_re(x_img) + self.conv_im(x_real)
        output = torch.cat((real, imaginary), dim=1)
        return output


class Complex_resunit(nn.Module):
    def __init__(self, input_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0):
        super(Complex_resunit, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2
        self.bn1 = nn.BatchNorm1d(2 * self.input_channels)
        self.conv1 = ComplexConv(self.input_channels, self.out_channels, self.kernel_size, padding=self.padding)
        self.bn2 = nn.BatchNorm1d(2 * self.out_channels)
        self.conv2 = ComplexConv(self.out_channels, self.out_channels, self.kernel_size, padding=self.padding)
        self.conv3 = ComplexConv(self.out_channels, self.out_channels, 1, padding=0, stride=self.stride)

    def forward(self, x):
        y = self.bn1(x)
        y = F.relu(y)
        y = self.conv1(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.input_channels != self.out_channels or self.stride != 1:
            m = self.conv3(x)
        else:
            m = x
        combine = torch.add(y, m)
        return combine


class Complex_resblock(nn.Module):
    def __init__(self, num, input_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0):
        super(Complex_resblock, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num = num

        self.conv = ComplexConv(self.input_channels, self.out_channels, 1, padding=self.padding)
        self.mp = nn.MaxPool1d(kernel_size=4, stride=4)
        self.res_units = nn.ModuleList(
            [Complex_resunit(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding) for _
             in range(self.num)])

    def forward(self, x):
        y = self.conv(x)
        for res_unit in self.res_units:
            y = res_unit(y)
            y = self.mp(y)
        return y


class complex_branches(nn.Module):
    def __init__(self, num, kernel_size=3):
        # num代表每个卷集块重复的次数
        super(complex_branches, self).__init__()
        self.num = num
        self.kernel_size = kernel_size
        self.block1 = Complex_resblock(self.num, 1, 8, self.kernel_size)
        self.block2 = Complex_resblock(self.num, 8, 16, self.kernel_size)
        #self.block3 = Complex_resblock(self.num, 16, 32, self.kernel_size)
        #self.block4 = Complex_resblock(self.num, 32, 64, self.kernel_size)

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        #y = self.block3(y)
        #y = self.block4(y)
        return y

class CVSRN(nn.Module):
    def __init__(self, c_in, c_out):
        super(CVSRN, self).__init__()
        self.c_out = c_out
        self.branch1 = complex_branches(1, 3)
        self.branch2 = complex_branches(1, 5)
        self.branch3 = complex_branches(1, 7)
        self.linear1 = nn.LazyLinear(256)
        self.dr1 = nn.AlphaDropout(0.5)

        self.linear2 = nn.LazyLinear(256)
        self.dr2 = nn.AlphaDropout(0.5)

        self.linear3 = nn.LazyLinear(self.c_out)

    def forward(self, x):
        b1 = self.branch1(x[:, :, :x.size()[2] // 3])
        b2 = self.branch2(x[:, :, x.size()[2] // 3:2 * (x.size()[2] // 3)])
        b3 = self.branch3(x[:, :, 2 * (x.size()[2] // 3):])
        y = torch.cat([b1, b2, b3], axis=2)
        y = torch.flatten(y, 1)
        y = self.linear1(y)
        y = F.selu(y)
        y = self.dr1(y)
        y = self.linear2(y)
        y = F.selu(y)
        embedding_output = self.dr2(y)
        cls_output = self.linear3(embedding_output)
        return embedding_output, cls_output


if __name__ == '__main__':
    x = torch.rand((1, 2, 6000), dtype=torch.float32).cuda()
    model = CVSRN(2, 16).cuda()
    y = model(x)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (2, 6000), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {macs}")
    print(f"Params: {params}")
