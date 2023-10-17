from torch import nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpSampler(nn.Module):
    """Up-scaling then double conv"""

    def __init__(self, in_channels, out_channels, mode="conv", scale_factor=2):
        super().__init__()
        h_channels = (in_channels + out_channels) // 2
        if mode in ["conv", "convolution"]:
            self.up = nn.ConvTranspose2d(in_channels, h_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(h_channels, out_channels)
        else:
            # if {bilinear, nearest,..}, use the normal convolutions to reduce the number of channels
            align_corners = None if mode == "nearest" else True  # align_corners does not work for nearest neighbor
            self.up = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)
            self.conv = DoubleConv(in_channels, out_channels, h_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # (B, C_in, H_out, W_out)
        x1 = self.conv(x1)
        # (B, C_out, H_out, W_out)
        return x1
