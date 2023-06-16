import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Conv2dRe(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, bias=True):
        super(Conv2dRe, self).__init__()
        self.bcr = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bcr(x)


class ADFM(nn.Module):

    def __init__(self, channels0, channels1, channels2, channels3, channels4):
        super(ADFM, self).__init__()

        channels = [channels4, channels3, channels2, channels1, channels0]

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.convl4 = nn.Sequential(
            Conv2dRe(channels[3] * 2, channels[3]),
            Conv2dRe(channels[3], channels[3])
        )

        self.conv5 = nn.Sequential(
            Conv2dRe(channels[4], channels[4]),
            Conv2dRe(channels[4], channels[3]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.conv4 = nn.Sequential(
            Conv2dRe(channels[3] * 2, channels[3]),
            Conv2dRe(channels[3], channels[3]),
            Conv2dRe(channels[3], channels[2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.conv3 = nn.Sequential(
            Conv2dRe(channels[2] * 2, channels[2]),
            Conv2dRe(channels[2], channels[2]),
            Conv2dRe(channels[2], channels[1]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.conv2 = nn.Sequential(
            Conv2dRe(channels[1] * 2, channels[1]),
            Conv2dRe(channels[1], channels[0]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.cr1 = nn.Sequential(
            nn.Conv2d(channels[4], channels[3], kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(channels[3], channels[3], kernel_size=3, padding=1)
        )

        self.cr2 = nn.Sequential(
            Conv2dRe(channels[4] + channels[3] + channels[2], channels[2], kernel_size=1, padding=0),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1)
        )

        self.cr3 = nn.Sequential(
            Conv2dRe(channels[4] + channels[3] + channels[2] + channels[1], channels[1],kernel_size=1, padding=0),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1)
        )

        self.conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.conv11 = Conv2dRe(channels[3], channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv12 = Conv2dRe(channels[2], channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv13 = Conv2dRe(channels[1], channels[1], kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x1, x2, x3, x4):
        dp5 = self.conv5(x1)

        block4 = self.cr1(self.upsample(x1))
        yres4 = self.convl4(torch.cat((block4, x2), dim=1))
        yfin4 = self.conv11(torch.mul(yres4, dp5) + dp5)
        d4put = torch.cat([dp5, yfin4], dim=1)
        dp4 = self.conv4(d4put)

        yres3 = self.cr2(torch.cat([self.upsample4(x1),self.upsample(x2),x3], dim=1))
        yfin3 = self.conv12(torch.mul(yres3, dp4) + dp4)
        d3put = torch.cat([dp4, yfin3], dim=1)
        dp3 = self.conv3(d3put)

        yres2 = self.cr3(torch.cat([self.upsample8(x1),self.upsample4(x2),self.upsample(x3),x4], dim=1))
        yfin2 = self.conv13(torch.mul(yres2, dp3) + dp3)
        d2put = torch.cat([dp3, yfin2], dim=1)
        dp2 = self.conv2(d2put)

        fin = self.conv(dp2)

        return fin
