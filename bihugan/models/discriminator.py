import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        layers = []
        ch_in = in_ch
        chs = [base * (2 ** i) for i in range(5)]
        for i, ch_out in enumerate(chs):
            padding = 1 if i < 4 else 2
            layers.extend([
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=padding, bias=False),
                nn.InstanceNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            ch_in = ch_out
        layers.append(nn.Conv2d(ch_in, 1, kernel_size=4, stride=1, padding=1, bias=True))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)