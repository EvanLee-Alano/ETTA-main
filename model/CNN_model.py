import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_outdoor(nn.Module):
    def __init__(self, in_channels=2, num_classes=7):
        super(CNN_outdoor, self).__init__()
        self.in_channel = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels=self.in_channel, out_channels=19, kernel_size=64, stride=8, padding=28)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=19, out_channels=15, kernel_size=8, stride=2, padding=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=15, out_channels=11, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.dense1 = nn.LazyLinear(out_features=128)
        self.drop1 = nn.Dropout(p=0.5)

        self.dense2 = nn.LazyLinear(out_features=16)
        self.drop2 = nn.Dropout(p=0.5)

        self.dense3 = nn.LazyLinear(out_features=self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.elu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.elu(x)
        x = self.maxpool3(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = F.elu(x)
        x = self.drop1(x)

        x = self.dense2(x)
        embedding_output = F.elu(x)

        cls_output = self.dense3(self.drop2(embedding_output))

        return embedding_output, cls_output


class CNN_lab(nn.Module):
    def __init__(self, in_channels=2, num_classes=7):
        super(CNN_lab, self).__init__()
        self.in_channel = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels=self.in_channel, out_channels=19, kernel_size=64, stride=8, padding=28)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(19)

        self.conv2 = nn.Conv1d(in_channels=19, out_channels=15, kernel_size=8, stride=2, padding=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.bn2 = nn.BatchNorm1d(15)

        self.flatten = nn.Flatten()

        self.dense1 = nn.LazyLinear(out_features=128)
        self.drop1 = nn.Dropout(p=0.5)

        self.dense2 = nn.LazyLinear(out_features=16)
        self.drop2 = nn.Dropout(p=0.5)

        self.dense3 = nn.LazyLinear(out_features=self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x)
        x = self.maxpool1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.elu(x)
        x = self.maxpool2(x)
        x = self.bn2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = F.elu(x)
        x = self.drop1(x)

        x = self.dense2(x) 
        embedding_output = F.elu(x)

        cls_output = self.dense3(self.drop2(embedding_output))

        return embedding_output, cls_output


if __name__ == '__main__':
    x = torch.rand((1, 2, 6000), dtype=torch.float32).cuda()
    model = CNN_outdoor(2, 16).cuda()
    y = model(x)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (2, 6000), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {macs}")
    print(f"Params: {params}")
