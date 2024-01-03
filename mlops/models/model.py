from torch import nn

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(p=0.55)

        self.activation = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = x.view(x.shape[0], -1)

        x = self.activation(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn6(self.fc3(x)))
        x = self.logsoftmax(self.fc4(x))

        return x
