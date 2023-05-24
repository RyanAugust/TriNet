import torch
import torch.nn as nn

class PerformanceCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, input_shape=(20, 50)) ## wut?
        self.relu1 = nn.ReLU()

        # Pooling layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Pooling layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer 1
        self.fc1 = nn.Linear(64 * 10 * 25, 128)
        self.relu3 = nn.ReLU()

        # Fully connected layer 2
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 10 * 25)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
    
