import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, c1: int, c2: tuple, c3: tuple, c4: int):
        super().__init__()
        self.branch_1 = ConvBlock(in_channels, c1, kernel_size=1)
        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels, c2[0], kernel_size=1),
            ConvBlock(c2[0], c2[1], kernel_size=3, padding=1)
        )
        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels, c3[0], kernel_size=1),
            ConvBlock(c3[0], c3[1], kernel_size=5, padding=2)
        )
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            ConvBlock(in_channels, c4, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch_1(x),
            self.branch_2(x),
            self.branch_3(x),
            self.branch_4(x)
        ], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.conv_1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pooling_1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv_2 = ConvBlock(64, 64, kernel_size=1)
        self.conv_3 = ConvBlock(64, 192, kernel_size=3, padding=1)
        self.pooling_2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception_3a = InceptionBlock(192, 64, (96, 128), (16, 32), 32)
        self.inception_3b = InceptionBlock(256, 128, (128, 192), (32, 96), 64)
        self.pooling_3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception_4a = InceptionBlock(480, 192, (96, 208), (16, 48), 64)
        self.inception_4b = InceptionBlock(512, 160, (112, 224), (24, 64), 64)
        self.inception_4c = InceptionBlock(512, 128, (128, 256), (24, 64), 64)
        self.inception_4d = InceptionBlock(512, 112, (144, 288), (32, 64), 64)
        self.inception_4e = InceptionBlock(528, 256, (160, 320), (32, 128), 128)
        self.pooling_4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception_5a = InceptionBlock(832, 256, (160, 320), (32, 128), 128)
        self.inception_5b = InceptionBlock(832, 384, (192, 384), (48, 128), 128)

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(images)          # (bs, 64, 112, 112)
        x = self.pooling_1(x)            # (bs, 64, 56, 56)

        x = self.conv_2(x)               # (bs, 64, 56, 56)
        x = self.conv_3(x)               # (bs, 192, 56, 56)
        x = self.pooling_2(x)            # (bs, 192, 28, 28)

        x = self.inception_3a(x)         # (bs, 256, 28, 28)
        x = self.inception_3b(x)         # (bs, 480, 28, 28)
        x = self.pooling_3(x)            # (bs, 480, 14, 14)

        x = self.inception_4a(x)         # (bs, 512, 14, 14)
        x = self.inception_4b(x)         # (bs, 512, 14, 14)
        x = self.inception_4c(x)         # (bs, 512, 14, 14)
        x = self.inception_4d(x)         # (bs, 528, 14, 14)
        x = self.inception_4e(x)         # (bs, 832, 14, 14)
        x = self.pooling_4(x)            # (bs, 832, 7, 7)

        x = self.inception_5a(x)         # (bs, 832, 7, 7)
        x = self.inception_5b(x)         # (bs, 1024, 7, 7)

        x = self.avg_pooling(x).flatten(1)  # (bs, 1024)
        return self.fc(self.dropout(x))     # (bs, num_classes)
