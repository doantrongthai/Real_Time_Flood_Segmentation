import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymConvBlock(nn.Module):
    def __init__(self, in_c, n_filters=128, kernel_size=3, dilation_rate=1):
        super().__init__()
        ks = kernel_size
        pad_h = dilation_rate * (ks // 2) if ks > 1 else 0
        pad_w = dilation_rate * (ks // 2) if ks > 1 else 0
        self.conv1 = nn.Conv2d(in_c, n_filters, kernel_size=(ks, 1),
                               padding=(pad_h, 0), dilation=dilation_rate, bias=False)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=(1, ks),
                               padding=(0, pad_w), dilation=dilation_rate, bias=False)
        self.bn = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))), inplace=True)


class EDRBSub(nn.Module):
    def __init__(self, in_c, h, dilation_rate):
        super().__init__()
        self.pw    = nn.Conv2d(in_c, h, 1, bias=True)
        self.bn_pw = nn.BatchNorm2d(h)
        self.dw1   = nn.Conv2d(h, h, kernel_size=(3, 1), padding=(1, 0), groups=h, bias=True)
        self.dw2   = nn.Conv2d(h, h, kernel_size=(1, 3), padding=(0, 1), groups=h, bias=True)
        self.bn_dw = nn.BatchNorm2d(h)
        self.co1   = nn.Conv2d(h, h, kernel_size=(3, 1),
                               padding=(dilation_rate, 0), dilation=dilation_rate, bias=True)
        self.co2   = nn.Conv2d(h, h, kernel_size=(1, 3),
                               padding=(0, dilation_rate), dilation=dilation_rate, bias=True)
        self.bn_co  = nn.BatchNorm2d(h)
        self.pw2    = nn.Conv2d(h * 2, h, 1, bias=True)
        self.bn_pw2 = nn.BatchNorm2d(h)

    def forward(self, x):
        x1  = F.relu(self.bn_pw(self.pw(x)), inplace=True)
        dw  = F.relu(self.bn_dw(self.dw2(self.dw1(x1))), inplace=True)
        co  = F.relu(self.bn_co(self.co2(self.co1(x1))), inplace=True)
        x2  = F.relu(self.bn_pw2(self.pw2(torch.cat([dw, co], dim=1))), inplace=True)
        return x1 + x2


class EDRB(nn.Module):
    def __init__(self, in_c, n_filters, dilation_rate=1):
        super().__init__()
        h = n_filters // 2
        self.block1 = EDRBSub(in_c, h, dilation_rate)
        self.block2 = EDRBSub(h,    h, dilation_rate)

    def forward(self, x):
        x1   = self.block1(x)
        x2   = self.block2(x1)
        return x1 + x2


class EHAAC(nn.Module):
    def __init__(self, in_c, n_filters):
        super().__init__()
        self.out1  = nn.Conv2d(in_c, n_filters, 1, bias=True)
        self.out6  = AsymConvBlock(in_c, n_filters, kernel_size=3, dilation_rate=6)
        self.out12 = AsymConvBlock(in_c, n_filters, kernel_size=3, dilation_rate=12)
        self.out18 = AsymConvBlock(in_c, n_filters, kernel_size=3, dilation_rate=18)

        self.sig1  = nn.Conv2d(n_filters, 1, 5, padding=2, bias=False)
        self.sig6  = nn.Conv2d(n_filters, 1, 5, padding=2, bias=False)
        self.sig12 = nn.Conv2d(n_filters, 1, 5, padding=2, bias=False)
        self.sig18 = nn.Conv2d(n_filters, 1, 5, padding=2, bias=False)

        cam_c = n_filters * 4
        self.cam_fc1    = nn.Linear(cam_c, cam_c)
        self.cam_fc2    = nn.Linear(cam_c, cam_c)
        self.final_conv = AsymConvBlock(cam_c, 128, kernel_size=1)

    def forward(self, x):
        o1  = self.out1(x)
        o6  = self.out6(x)
        o12 = self.out12(x)
        o18 = self.out18(x)

        con1 = torch.cat([o1, o6, o12, o18], dim=1)

        m1  = o1  * torch.sigmoid(self.sig1(o1))
        m6  = o6  * torch.sigmoid(self.sig6(o6))
        m12 = o12 * torch.sigmoid(self.sig12(o12))
        m18 = o18 * torch.sigmoid(self.sig18(o18))
        con2 = torch.cat([m1, m6, m12, m18], dim=1)

        B, C, _, _ = con2.shape
        cam = con2.mean(dim=(2, 3))
        cam = torch.sigmoid(self.cam_fc2(self.cam_fc1(cam)))
        cam = cam.view(B, C, 1, 1)

        out = con1 + con2 * cam
        return self.final_conv(out)


class FASegNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.enc1 = EDRB(in_channels, 24,  dilation_rate=2)
        self.enc2 = EDRB(12,          48,  dilation_rate=4)
        self.enc3 = EDRB(24,          72,  dilation_rate=8)
        self.enc4 = EDRB(36,          96,  dilation_rate=16)

        self.bottleneck = EHAAC(48, 72)

        self.up6   = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec6  = EDRB(144, 96, dilation_rate=16)

        self.up7   = nn.ConvTranspose2d(48, 72, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec7  = EDRB(108, 72, dilation_rate=8)

        self.up8   = nn.ConvTranspose2d(36, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec8  = EDRB(72, 48, dilation_rate=4)

        self.up9   = nn.ConvTranspose2d(24, 24, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec9  = EDRB(36, 24, dilation_rate=2)

        self.head  = nn.Conv2d(12, num_classes, kernel_size=1)

    def forward(self, x):
        f1 = self.enc1(x)
        p1 = F.max_pool2d(f1, 2)

        f2 = self.enc2(p1)
        p2 = F.max_pool2d(f2, 2)

        f3 = self.enc3(p2)
        p3 = F.max_pool2d(f3, 2)

        f4 = self.enc4(p3)
        p4 = F.max_pool2d(f4, 2)

        bn = self.bottleneck(p4)

        d6 = self.dec6(torch.cat([self.up6(bn), f4], dim=1))
        d7 = self.dec7(torch.cat([self.up7(d6), f3], dim=1))
        d8 = self.dec8(torch.cat([self.up8(d7), f2], dim=1))
        d9 = self.dec9(torch.cat([self.up9(d8), f1], dim=1))

        return self.head(d9)


def build_model(num_classes=1):
    return FASegNet(in_channels=3, num_classes=num_classes)
