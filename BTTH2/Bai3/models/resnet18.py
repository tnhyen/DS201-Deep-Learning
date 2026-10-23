import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn_1(self.conv_1(x)))
        y = self.bn_2(self.conv_2(y))
        return F.relu(y + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_1 = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
        self.pooling_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.layer_2 = nn.Sequential(ResidualBlock(64, 128), ResidualBlock(128, 128))
        self.pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.layer_3 = nn.Sequential(ResidualBlock(128, 256), ResidualBlock(256, 256))
        self.pooling_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.layer_4 = nn.Sequential(ResidualBlock(256, 512), ResidualBlock(512, 512))

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_1(self.conv_1(images)))  # (bs, 64, 112, 112)
        x = self.pooling_1(x)                        # (bs, 64, 56, 56)

        x = self.layer_1(x)                          # (bs, 64, 56, 56)
        x = self.pooling_2(x)                        # (bs, 64, 27, 27)

        x = self.layer_2(x)                          # (bs, 128, 27, 27)
        x = self.pooling_3(x)                        # (bs, 128, 13, 13)

        x = self.layer_3(x)                          # (bs, 256, 13, 13)
        x = self.pooling_4(x)                        # (bs, 256, 6, 6)

        x = self.layer_4(x)                          # (bs, 512, 6, 6)

        x = self.avg_pooling(x).flatten(1)           # (bs, 512)
        return self.fc(x)                            # (bs, num_classes)
