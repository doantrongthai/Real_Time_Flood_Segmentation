import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, nn.ReLU(inplace=True), resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.layer0(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        return low, x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.adapool = nn.AdaptiveAvgPool2d(1)

        self.convf = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, padding=0, bias=False)
        self.bnf = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))

        x5 = self.adapool(x)
        x5 = self.relu(self.bn5(self.conv5(x5)))
        x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.relu(self.bnf(self.convf(x)))
        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()

        self.backbone = ResNet50()
        self.aspp = ASPP(in_channels=1024, out_channels=256)

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.shape

        low, x = self.backbone(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=True)

        low = self.low_level_conv(low)

        x = torch.cat([low, x], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


def build_model(num_classes=21):
    return DeepLabV3Plus(num_classes=num_classes)