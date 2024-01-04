from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, c1out=32, c2out=64, c3out=64, fc1out=1028, fc2out=128, fc3out=64, p_drop=0.55):
        super().__init__()
        self.fc1 = nn.Linear(c3out * 28 * 28, fc1out)
        self.fc2 = nn.Linear(fc1out, fc2out)
        self.fc3 = nn.Linear(fc2out, fc3out)
        self.fc4 = nn.Linear(fc3out, 10)

        self.conv1 = nn.Conv2d(1, c1out, 3, padding=1)
        self.conv2 = nn.Conv2d(c1out, c2out, 3, padding=1)
        self.conv3 = nn.Conv2d(c2out, c3out, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(c1out)
        self.bn2 = nn.BatchNorm2d(c2out)
        self.bn3 = nn.BatchNorm2d(c3out)
        self.bn4 = nn.BatchNorm1d(fc1out)
        self.bn5 = nn.BatchNorm1d(fc2out)
        self.bn6 = nn.BatchNorm1d(fc3out)

        self.dropout = nn.Dropout(p=p_drop)

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
