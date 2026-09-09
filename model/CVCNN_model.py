import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Function
#from utils.complexcnn import ComplexConv
#CVCNN

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


class CVCNN(nn.Module):
    def __init__(self, in_channels=2, num_classes=16, channels=64):
        super(CVCNN, self).__init__()
        self.in_channel = in_channels
        self.num_classes = num_classes
        self.channel = channels

        self.conv1 = ComplexConv(in_channels=int(self.in_channel/2),out_channels=self.channel,kernel_size=3)
        self.batchnorm1 = nn.BatchNorm1d(num_features=int(self.channel*2))
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = ComplexConv(in_channels=self.channel,out_channels=self.channel,kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(num_features=int(self.channel*2))
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = ComplexConv(in_channels=self.channel, out_channels=self.channel, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm1d(num_features=int(self.channel*2))
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = ComplexConv(in_channels=self.channel, out_channels=self.channel, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm1d(num_features=int(self.channel*2))
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = ComplexConv(in_channels=self.channel, out_channels=self.channel, kernel_size=3)
        self.batchnorm5 = nn.BatchNorm1d(num_features=int(self.channel*2))
        self.maxpool5 = nn.MaxPool1d(kernel_size=2)
        self.conv6 = ComplexConv(in_channels=self.channel, out_channels=self.channel, kernel_size=3)
        self.batchnorm6 = nn.BatchNorm1d(num_features=int(self.channel*2))
        self.maxpool6 = nn.MaxPool1d(kernel_size=2)
        self.conv7 = ComplexConv(in_channels=self.channel, out_channels=self.channel, kernel_size=3)
        self.batchnorm7 = nn.BatchNorm1d(num_features=int(self.channel*2))
        self.maxpool7 = nn.MaxPool1d(kernel_size=2)
        self.conv8 = ComplexConv(in_channels=self.channel, out_channels=self.channel, kernel_size=3)
        self.batchnorm8 = nn.BatchNorm1d(num_features=int(self.channel*2))
        self.maxpool8 = nn.MaxPool1d(kernel_size=2)
        self.conv9 = ComplexConv(in_channels=self.channel, out_channels=self.channel, kernel_size=3)
        self.batchnorm9 = nn.BatchNorm1d(num_features=int(self.channel*2))
        self.maxpool9 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(1024)

        self.linear2 = nn.LazyLinear(self.num_classes)


    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.maxpool5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.maxpool6(x)
        x = F.relu(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.maxpool7(x)
        x = F.relu(x)

        x = self.conv8(x)
        x = self.batchnorm8(x)
        x = self.maxpool8(x)
        x = F.relu(x)

        x = self.conv9(x)
        x = self.batchnorm9(x)
        x = self.maxpool9(x)
        x = F.relu(x)

        x = self.flatten(x)

        x = self.linear1(x)
        embedding_output = F.relu(x)

        cls_output = self.linear2(embedding_output)
        return embedding_output, cls_output


if __name__ == '__main__':
    x = torch.rand((1, 2, 6000), dtype=torch.float32).cuda()
    model = CVCNN(2, num_classes=16).cuda()
    y = model(x)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (2, 6000), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {macs}")
    print(f"Params: {params}")
