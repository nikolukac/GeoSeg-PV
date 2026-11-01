import torch
from torch import nn


class FineOptimizationModule(nn.Module):
    """Fine Optimization Module (FOM) that refines the logits of a segmentation network.

    A small U-shaped network after PVNet (Wang et al., Int. J. Appl. Earth Obs. Geoinf. 119 (2023) 103309):
    4 encoder stages (3x3 conv, BN, ReLU, 2x2 max pooling), a bridge, and 4 decoder stages
    (3x3 transposed conv, BN, ReLU) with skip connections. The input size must be divisible by 16.
    """

    def __init__(self, input_channels=2, channels=64):
        super(FineOptimizationModule, self).__init__()

        def down(in_ch):
            return nn.Sequential(nn.Conv2d(in_ch, channels, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, stride=2))

        def up(in_ch):
            return nn.Sequential(nn.ConvTranspose2d(in_ch, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.BatchNorm2d(channels),
                                 nn.ReLU(inplace=True))

        self.encoder1 = down(input_channels)
        self.encoder2 = down(channels)
        self.encoder3 = down(channels)
        self.encoder4 = down(channels)
        self.bridge = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(channels),
                                    nn.ReLU(inplace=True))
        self.decoder4 = up(channels)
        self.decoder3 = up(2 * channels)
        self.decoder2 = up(2 * channels)
        self.decoder1 = up(2 * channels)
        self.output_layer = nn.Conv2d(channels, input_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)  # 1/2
        e2 = self.encoder2(e1)  # 1/4
        e3 = self.encoder3(e2)  # 1/8
        e4 = self.encoder4(e3)  # 1/16
        b = self.bridge(e4)
        d4 = torch.cat((self.decoder4(b), e3), dim=1)  # 1/8
        d3 = torch.cat((self.decoder3(d4), e2), dim=1)  # 1/4
        d2 = torch.cat((self.decoder2(d3), e1), dim=1)  # 1/2
        d1 = self.decoder1(d2)  # 1/1
        return self.output_layer(d1)
