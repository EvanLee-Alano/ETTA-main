import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import MaxPool1d, MaxPool2d, Conv2d, Conv1d, Flatten, ReLU, LazyLinear, Dropout
from torch.autograd import Function
import math

class SPPLayer(nn.Module):

    def __init__(self, size):
        super(SPPLayer, self).__init__()
        self.size = size

    def forward(self, x):
        num, c, h, w = x.size()
        stride = int(math.ceil(w / self.size))
        padding = int((stride*self.size - w + 1)/2)
        x = F.pad(x, (padding, padding))
        tensor = F.max_pool2d(x, kernel_size=(1, stride), stride=(1, stride))
        return tensor


class MSCNN(nn.Module):
    def __init__(self, c_in=2, c_out=10, out_channels=128, spp_output=16):
        super(MSCNN, self).__init__()
        self.c_in = c_in
        self.out_channels = out_channels
        self.c_out = c_out
        self.spp_output = spp_output

        self.down_sampling1 = Conv1d(self.c_in, self.c_in, 1, stride=2, padding=0, bias = False)
        self.down_sampling2 = Conv1d(self.c_in, self.c_in, 1, stride=4, padding=0, bias = False)
        self.conv_layer = nn.Sequential(
            Conv2d(1, self.out_channels, kernel_size=(1, 10), padding=(0, 4)),
            ReLU(),
            MaxPool2d(3, padding=1),
            Conv2d(self.out_channels, self.out_channels * 2, kernel_size=(2, 3), padding=(1, 1)),
            ReLU(),
        )

        self.conv_layer1 = nn.Sequential(
            Conv2d(1, self.out_channels, kernel_size=(1, 10), padding=(0, 4)),
            ReLU(),
            MaxPool2d(3, padding=1),
            Conv2d(self.out_channels, self.out_channels * 2, kernel_size=(2, 3), padding=(1, 1)),
            ReLU(),
        )

        self.conv_layer2 = nn.Sequential(
            Conv2d(1, self.out_channels, kernel_size=(1, 10), padding=(0, 4)),
            ReLU(),
            MaxPool2d(3, padding=1),
            Conv2d(self.out_channels, self.out_channels * 2, kernel_size=(2, 3), padding=(1, 1)),
            ReLU(),
        )
        self.spp = SPPLayer(self.spp_output)

        self.head = nn.Sequential(
            Flatten(),
            LazyLinear(self.out_channels),
            ReLU(),
            Dropout(0.5),
            )

        self.cls_head = LazyLinear(self.c_out)

    def forward(self, x):
        x1 = self.down_sampling1(x)
        x2 = self.down_sampling2(x)

        x = torch.unsqueeze(x, dim=1)
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)

        x = self.conv_layer(x)
        x1 = self.conv_layer1(x1)
        x2 = self.conv_layer2(x2)
        
        x = self.spp(x)
        x1 = self.spp(x1)
        x2 = self.spp(x2)

        embedding_output = self.head(torch.cat([x, x1, x2], dim=2))
        cls_output = self.cls_head(embedding_output)

        return embedding_output, cls_output


if __name__ == '__main__':
    x = torch.rand((1, 2, 6000), dtype=torch.float32).cuda()
    model = MSCNN(2, 16).cuda()
    y = model(x)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (2, 6000), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {macs}")
    print(f"Params: {params}")
