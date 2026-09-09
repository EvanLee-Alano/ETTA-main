#python = 3.8.18, torch = 2.0.0
import torch
import torch.nn as nn


import torch
import torch.nn as nn

class SeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super(SeparableConv1D, self).__init__()

        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, groups=groups)
        self.bn1 = nn.BatchNorm1d(in_channels)

        # Pointwise convolution
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.bn1(x)

        # Pointwise convolution
        x = self.pointwise_conv(x)
        x = self.bn2(x)

        return x

class LightCNN(nn.Module):
    def __init__(self,in_channels=2, channels=64, num_classes=121):
        super(LightCNN, self).__init__()
        self.in_channel = in_channels
        self.num_classes = num_classes
        self.out_channel = channels

        # First layer: Conv1D(128, 1x8) + ReLU + Dropout(0.5)
        self.conv1 = nn.Conv1d(in_channels=self.in_channel, out_channels=128, kernel_size=8)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        # Second layer: SeparableConv1D(128, 1x8) + ReLU + Dropout(0.5)
        self.sep_conv = SeparableConv1D(in_channels=128, out_channels=128, kernel_size=8, groups=128)  # SeparableConv1D
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        # Third layer: Dense(4) + Softmax
        self.dense = nn.Linear(128, 4)

        self.linear2 = nn.LazyLinear(self.num_classes)


    def forward(self, x):
        # First layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Second layer
        x = self.sep_conv(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        embedding_output = x

        # Global average pooling
        x = torch.mean(x, dim=2)

        # Third layer
        x = self.dense(x)
        cls_output =  self.linear2(x)

        return embedding_output, cls_output


if __name__ == '__main__':
    x = torch.rand((1, 2, 6000), dtype=torch.float32).cuda()
    model = LightCNN(2, num_classes=16).cuda()
    y = model(x)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (2, 6000), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {macs}")
    print(f"Params: {params}")
