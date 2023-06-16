import torch
import torch.nn as nn
import torch.nn.functional as F


class GDM(nn.Module):
    def __init__(self, channels):
        super(GDM, self).__init__()
        self.convk1d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0,
                      groups=channels, bias=False)
        )
        self.convk3d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0),
                      groups=channels, bias=False)
        )
        self.convk5d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 5), stride=1, padding=(0, 2),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(5, 1), stride=1, padding=(2, 0),
                      groups=channels, bias=False)
        )
        self.convk7d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 7), stride=1, padding=(0, 3),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(7, 1), stride=1, padding=(3, 0),
                      groups=channels, bias=False)
        )

    def forward(self, x):
        c1 = self.convk1d1(x)
        c2 = self.convk3d1(x)
        c3 = self.convk5d1(x)
        c4 = self.convk7d1(x)
        return c1, c2, c3, c4


class GVM(nn.Module):
    def __init__(self, channels):
        super(GVM, self).__init__()
        self.convk1d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0,
                      groups=channels, bias=False)
        )
        self.convk3d3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 3), dilation=(1, 3),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(3, 0), dilation=(3, 1),
                      groups=channels, bias=False)
        )
        self.convk3d5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 5), dilation=(1, 5),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(5, 0), dilation=(5, 1),
                      groups=channels, bias=False)
        )
        self.convk3d7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 7), dilation=(1, 7),
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(7, 0), dilation=(7, 1),
                      groups=channels, bias=False)
        )
        self.conv1 = nn.Conv2d(channels*4, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.convk1 = nn.Conv2d(channels, channels // 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, a0, a1, a2, a3, a):
        c1 = self.convk1d1(x)
        c5 = self.convk3d3(x)
        c6 = self.convk3d5(x)
        c7 = self.convk3d7(x)
        y1 = self.relu(c1 * a0 + c1 * a1 + c1 * a2 + c1 * a3)
        y2 = self.relu(c5 * a0 + c5 * a1 + c5 * a2 + c5 * a3)
        y3 = self.relu(c6 * a0 + c6 * a1 + c6 * a2 + c6 * a3)
        y4 = self.relu(c7 * a0 + c7 * a1 + c7 * a2 + c7 * a3)
        res = self.conv1(torch.cat((y1, y2, y3, y4), dim=1))
        res = self.relu(res * a)
        return self.convk1(res)


class CAM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CAM, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, out_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(fea))))
        out = max_out
        ress = self.sigmoid(out)
        return ress


class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv2d(out)
        out = self.sigmoid(out)
        return out


class CAFM(nn.Module):
    def __init__(self, channels):
        super(CAFM, self).__init__()
        self.ca = CAM(channels * 3, 1)
        self.sa = SAM()
        self.gdm = GDM(channels * 3)
        self.gvm = GVM(channels * 3)

    def forward(self, x1, x2):
        out = torch.cat((x1, x2, x1 + x2), dim=1)
        a = self.ca(out).view(-1)
        out0, out1, out2, out3 = self.gdm(out)
        out = self.gvm(out, out0, out1, out2, out3, a)
        out = self.sa(out) * out
        return out




