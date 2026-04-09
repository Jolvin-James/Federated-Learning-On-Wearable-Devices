# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HAR_CNN(nn.Module):
    def __init__(self, num_classes=6):
        super(HAR_CNN, self).__init__()

        self.conv1 = nn.Conv1d(9, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        self._to_linear = None

        # Placeholder (will reset later)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._forward_conv(x)

        if self._to_linear is None:
            self._to_linear = x.view(x.size(0), -1).shape[1]
            self.fc1 = nn.Linear(self._to_linear, 128).to(x.device)

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x