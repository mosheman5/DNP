import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self, down_in_channels=[2, 35, 70],
                 down_out_channels=[35, 70, 70],
                 up_in_channels=[140, 70],
                 up_out_channels=[35, 35],
                 kernel=5,
                 dilation=3,
                 pool_kernel=2):
        super(UNet, self).__init__()

        self.encoder1 = ConvBlock(down_in_channels[0], down_out_channels[0], kernel, dilation)
        self.encoder2 = ConvBlock(down_in_channels[1], down_out_channels[1], kernel, dilation)

        self.bottleneck = ConvBlock(down_in_channels[2], down_out_channels[2], kernel, dilation)
        self.decoder2 = ConvBlock(up_in_channels[0], up_out_channels[0], kernel, dilation)
        self.decoder1 = ConvBlock(up_in_channels[1], up_out_channels[1], kernel, dilation)

        self.pool = nn.AvgPool2d(pool_kernel)
        self.conv = nn.Conv2d(
            in_channels=up_out_channels[-1], out_channels=down_in_channels[0], kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))
        in_dec2 = F.interpolate(bottleneck, size=enc2.size()[2:], mode='bilinear')
        dec2 = self.decoder2(torch.cat((in_dec2, enc2), dim=1))
        in_dec1 = F.interpolate(dec2, size=enc1.size()[2:], mode='bilinear')
        dec1 = self.decoder1(torch.cat((in_dec1, enc1), dim=1))

        return self.conv(dec1)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, dilation):
        super(ConvBlock, self).__init__()

        self.convs1 = ConvDeform(in_channels, in_channels, kernel, dilation)
        self.convs2 = ConvDeform(in_channels, out_channels, kernel, dilation)

    def forward(self, x):
        x = self.convs1(x)
        x = self.convs2(x)
        return x


class ConvDeform(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, dilation):
        super(ConvDeform, self).__init__()
        pad_dialted = dilation * (kernel - 1) // 2
        pad_reg = (kernel - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, dilation=(dilation, 1),
                              padding=(pad_dialted, pad_reg))
        self.norm = nn.InstanceNorm2d(num_features=out_channels)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activ(x)
