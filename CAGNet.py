import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from models.MyModel.VGG import VGG16
from models.CAINet2.Fusion import CAFM
from models.CAINet2.Decoder import ADFM


class MAINet(nn.Module):
    def __init__(self):
        super(MAINet, self).__init__()

        self.vgg_r = VGG16()
        self.vgg_d = VGG16()
        self.channels = [64, 128, 256, 512, 512]

        self.case1 = CASAE(64)
        self.case2 = CASAE(128)
        self.case3 = CASAE(256)
        self.case4 = CASAE(512)
        self.case5 = CASAE(512)

        self.fea1 = CAFM(64)
        self.fea2 = CAFM(128)
        self.fea3 = CAFM(256)
        self.fea4 = CAFM(512)
        self.fea5 = CAFM(512)

        self.conv5 = nn.Conv2d(self.channels[4], self.channels[4], kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.adfm = ADFM(512, 512, 256, 128, 64)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, RGBT):
        image = RGBT[0]
        t = RGBT[1]
        dlist = []

        conv1_vgg_r = self.vgg_r.conv1(image)
        conv1_vgg_d = self.vgg_d.conv1(t)
        conv1r = self.case1(conv1_vgg_r)
        conv1d = self.case1(conv1_vgg_d)
        conv2_vgg_r = self.vgg_r.conv2(conv1_vgg_r + conv1d + conv1r)
        conv2_vgg_d = self.vgg_d.conv2(conv1_vgg_d + conv1d + conv1r)

        conv3_vgg_d_in = self.fea2(conv2_vgg_r, conv2_vgg_d)
        dlist.append(conv3_vgg_d_in)
        conv2r = self.case2(conv2_vgg_r)
        conv2d = self.case2(conv2_vgg_d)
        conv3_vgg_r = self.vgg_r.conv3(conv2_vgg_r + conv2d + conv2r)
        conv3_vgg_d = self.vgg_d.conv3(conv2_vgg_d + conv2d + conv2r)

        conv3_vgg_3 = self.fea3(conv3_vgg_r, conv3_vgg_d)
        dlist.append(conv3_vgg_3)
        conv3r = self.case3(conv3_vgg_r)
        conv3d = self.case3(conv3_vgg_d)
        conv4_vgg_r = self.vgg_r.conv4(conv3_vgg_r + conv3d + conv3r)
        conv4_vgg_d = self.vgg_d.conv4(conv3_vgg_d + conv3d + conv3r)

        conv4_vgg_4 = self.fea4(conv4_vgg_r, conv4_vgg_d)
        dlist.append(conv4_vgg_4)
        conv4r = self.case4(conv4_vgg_r)
        conv4d = self.case4(conv4_vgg_d)
        conv5_vgg_r = self.vgg_r.conv5(conv4_vgg_r + conv4d + conv4r)
        conv5_vgg_d = self.vgg_d.conv5(conv4_vgg_d + conv4d + conv4r)
        conv5r = self.case5(conv5_vgg_r)
        conv5d = self.case5(conv5_vgg_d)
        conv5_vgg_5 = self.fea5(conv5_vgg_r + conv5r + conv5d, conv5_vgg_d + conv5d)
        dlist.append(conv5_vgg_5)

        smap = self.adfm(dlist[3], dlist[2], dlist[1], dlist[0])

        smap = self.reg_layer(smap)
        # print(smap)

        return smap


class CAE(nn.Module):
    def __init__(self, channels):
        super(CAE, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
        self.conv1 = nn.Conv2d(1, 1, 7, padding=3, bias=False)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        y, _ = torch.max(x, dim=1, keepdim=True)
        y = self.conv1(y)
        ys = torch.sigmoid(y)
        x = torch.sigmoid(x) * residual
        xo = x * ys
        return xo


class CASAE(nn.Module):
    def __init__(self, in_planes):
        super(CASAE, self).__init__()

        self.cae = CAE(in_planes)

    def forward(self, data):
        fea = self.cae(data)

        return fea


if __name__ == '__main__':
    rgb = torch.randn(1, 3, 256, 256)
    depth = torch.randn(1, 3, 256, 256)
    model = MAINet()
    res = model([rgb, depth])
    print(res.shape)
