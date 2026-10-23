import torch
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pooling_1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pooling_2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.unsqueeze(1)  # (bs, 1, 28, 28)

        x = F.sigmoid(self.conv_1(x))  # (bs, 6, 28, 28)
        x = self.pooling_1(x)  # (bs, 6, 14, 14)

        x = F.sigmoid(self.conv_2(x))  # (bs, 16, 10, 10)
        x = self.pooling_2(x)  # (bs, 16, 5, 5)

        x = x.flatten(1)  # (bs, 400)
        x = F.sigmoid(self.fc_1(x))  # (bs, 120)
        x = F.sigmoid(self.fc_2(x))  # (bs, 84)
        return self.fc_3(x)  # (bs, num_classes)
