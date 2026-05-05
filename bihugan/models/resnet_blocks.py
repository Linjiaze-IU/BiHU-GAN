import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, channels, use_dropout=False, p=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(channels)
        self.dropout = nn.Dropout2d(p) if use_dropout else None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.in1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.in2(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return x + y