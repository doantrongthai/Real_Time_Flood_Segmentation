import torch
import torch.nn as nn
import timm


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class Attention(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super().__init__()
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.in_places  = in_places
        self.eps        = eps
        self.query_conv = nn.Conv2d(in_places, in_places // scale, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_places, in_places // scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_places, in_places,          kernel_size=1)

    def forward(self, x):
        batch_size, chnnels, height, width = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size,   -1, width * height)
        V = self.value_conv(x).view(batch_size,  -1, width * height)

        Q = l2_norm(Q).permute(-3, -1, -2)
        K = l2_norm(K)

        tailor_sum  = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum   = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum   = value_sum.expand(-1, chnnels, width * height)
        matrix      = torch.einsum("bmn, bcn->bmc", K, V)
        matrix_sum  = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)
        return (self.gamma * weight_value).contiguous()


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_chan)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AttentionEnhancementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv       = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = Attention(out_chan)
        self.bn_atten   = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        feat = self.conv(x)
        att  = self.conv_atten(feat)
        return self.bn_atten(att)


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet      = timm.create_model('swsl_resnet18', features_only=True, output_stride=32,
                                             out_indices=(2, 3, 4), pretrained=False)
        self.arm16       = AttentionEnhancementModule(256, 128)
        self.arm32       = AttentionEnhancementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg    = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32        = nn.Upsample(scale_factor=2., mode='bilinear', align_corners=False)
        self.up16        = nn.Upsample(scale_factor=2., mode='bilinear', align_corners=False)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        avg         = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg         = self.conv_avg(avg)

        feat32_arm  = self.arm32(feat32)
        feat32_sum  = feat32_arm + avg
        feat32_up   = self.conv_head32(self.up32(feat32_sum))

        feat16_arm  = self.arm16(feat16)
        feat16_sum  = feat16_arm + feat32_up
        feat16_up   = self.conv_head16(self.up16(feat16_sum))

        return feat16_up, feat32_up


class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1    = ConvBNReLU(3,  64,  ks=7, stride=2, padding=3)
        self.conv2    = ConvBNReLU(64, 64,  ks=3, stride=2, padding=1)
        self.conv3    = ConvBNReLU(64, 64,  ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        return x


class FeatureAggregationModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk    = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv_atten = Attention(out_chan)

    def forward(self, fsp, fcp):
        fcat       = torch.cat([fsp, fcp], dim=1)
        feat       = self.convblk(fcat)
        atten      = self.conv_atten(feat)
        feat_out   = torch.mul(feat, atten) + feat
        return feat_out


class ABCNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.cp       = ContextPath()
        self.sp       = SpatialPath()
        self.fam      = FeatureAggregationModule(256, 256)
        self.conv_out = nn.Sequential(
            ConvBNReLU(256, 256, ks=3, stride=1, padding=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feat_cp8, _ = self.cp(x)
        feat_sp     = self.sp(x)
        feat_fuse   = self.fam(feat_sp, feat_cp8)
        return self.conv_out(feat_fuse)


def build_model(num_classes=1):
    return ABCNet(num_classes=num_classes)
