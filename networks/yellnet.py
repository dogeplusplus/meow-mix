import torch
import torch.nn as nn


class YellNet(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5), stride=(2,1), padding=(0, 2))
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(2,1), padding=(0, 2))
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 5), stride=(2,1), padding=(0, 2))
        self.relu3 = nn.LeakyReLU(0.1)
        self.dropout3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(32, 16, kernel_size=(3, 5), stride=(2,1), padding=(0, 2))
        self.relu4 = nn.LeakyReLU(0.1)
        self.dropout4 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout3(x)


        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.sigmoid(x)
        return x
