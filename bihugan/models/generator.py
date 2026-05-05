import torch
import torch.nn as nn
from .resnet_blocks import ResnetBlock
from .cbam import CBAM

class BiHUGenerator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64, n_res=9,
                 cbam_reduction=16, mc_dropout_p=0.0):
        super().__init__()
        use_dropout = mc_dropout_p > 0
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_ch, base, kernel_size=7, padding=0, bias=False)
        self.in1 = nn.InstanceNorm2d(base)
        self.act = nn.ReLU(inplace=True)

        self.down1 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base*2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base*4),
            nn.ReLU(inplace=True),
        )

        self.cbam1 = CBAM(base, reduction=cbam_reduction)
        self.cbam2 = CBAM(base*2, reduction=cbam_reduction)
        self.cbam3 = CBAM(base*4, reduction=cbam_reduction)
        self.cbam4 = CBAM(base*2, reduction=cbam_reduction)
        self.cbam5 = CBAM(base, reduction=cbam_reduction)

        self.res = nn.Sequential(*[
            ResnetBlock(base*4, use_dropout=use_dropout, p=mc_dropout_p)
            for _ in range(n_res)
        ])

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(base*2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base*2, base, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
        )

        self.pad_out = nn.ReflectionPad2d(3)
        self.conv_out = nn.Conv2d(base, out_ch, kernel_size=7, padding=0)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.in1(x)
        x = self.act(x)
        x = self.cbam1(x)
        x = self.down1(x)
        x = self.cbam2(x)
        x = self.down2(x)
        x = self.cbam3(x)
        x = self.res(x)
        x = self.up1(x)
        x = self.cbam4(x)
        x = self.up2(x)
        x = self.cbam5(x)
        x = self.pad_out(x)
        x = self.conv_out(x)
        return torch.tanh(x)