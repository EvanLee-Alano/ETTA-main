import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_block(nn.Module):
    def __init__(self):
        super(CNN_block, self).__init__()
        self.conv1 = nn.Conv1d(128, 128, 7)
        self.conv2 = nn.Conv1d(128, 128, 5)
        self.maxpool = nn.MaxPool1d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        return x


class ORACLE_CNN(nn.Module):
    def __init__(self, in_channels=2, num_classes=16):
        super(ORACLE_CNN, self).__init__()
        self.in_channel = in_channels
        self.num_class = num_classes

        self.cnn_blocks = nn.Sequential(
            nn.Conv1d(self.in_channel, 128, 7),
            nn.Conv1d(128, 128, 5),
            nn.MaxPool1d(3)
        )

        for i in range(1): # 1
            self.cnn_blocks.append(
                CNN_block()
            )

        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, self.num_class)

    def forward(self, x):
        x = self.cnn_blocks(x)
        x = F.dropout(self.flatten(x), 0.5)
        embedding_output = self.fc1(x)
        x = F.dropout(embedding_output, 0.5)
        cls_output = self.fc2(x)
        return embedding_output, cls_output

if __name__ == '__main__':
    x = torch.rand((1, 2, 6000), dtype=torch.float32).cuda()
    model = ORACLE_CNN(2, 16).cuda()
    y = model(x)

    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (2, 6000), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {macs}")
    print(f"Params: {params}")
