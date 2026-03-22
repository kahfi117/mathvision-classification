import torch
import torch.nn as nn


class ConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(out_channels, out_channels * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.out_channels = out_channels * 2

    def forward(self, x):
        return self.block(x)


class ParallelCNN(nn.Module):
    def __init__(self, in_channels, num_classes, branch_kernels=(5, 7), base_channels=32, dropout=0.3):
        super().__init__()

        self.branch1 = ConvBranch(in_channels, base_channels, branch_kernels[0])
        self.branch2 = ConvBranch(in_channels, base_channels, branch_kernels[1])

        merged_channels = self.branch1.out_channels + self.branch2.out_channels

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(merged_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        x = torch.cat([b1, b2], dim=1)
        return self.head(x)