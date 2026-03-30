# =============================================================
# Ablation D — ENet + DSConv + MaxAvg  (Model K)
# Kết hợp Depthwise Separable Conv trong RegularBottleneck
# và dual-path shortcut (MaxPool + AvgPool) trong DownsamplingBottleneck.
# Không có Imp. CA.
# Đây là nền trực tiếp để so sánh với FloodENet (thêm Imp. CA).
# =============================================================

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _max_unpool(unpool_layer, x, indices, output_size):
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    out = unpool_layer(x, indices, output_size=output_size)
    torch.use_deterministic_algorithms(prev)
    return out


def _dws_conv(channels, kernel_size=3, padding=1, dilation=1):
    """Depthwise Separable Convolution (DW + PW) cho regular / dilated."""
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding,
                  dilation=dilation, groups=channels, bias=False),
        nn.BatchNorm2d(channels), nn.PReLU(),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(channels), nn.PReLU(),
    )


def _dws_conv_asymmetric(channels, kernel_size=5, padding=2):
    """DWS cho asymmetric factorized filter (5×1 và 1×5)."""
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                  padding=(padding, 0), groups=channels, bias=False),
        nn.BatchNorm2d(channels), nn.PReLU(),
        nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                  padding=(0, padding), groups=channels, bias=False),
        nn.BatchNorm2d(channels), nn.PReLU(),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(channels), nn.PReLU(),
    )


# ─────────────────────────────────────────────────────────────
# Blocks
# ─────────────────────────────────────────────────────────────

class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.main_branch = nn.Conv2d(
            in_channels, out_channels - in_channels,
            kernel_size=3, stride=2, padding=1, bias=False,
        )
        self.ext_branch     = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batch_norm     = nn.BatchNorm2d(out_channels)
        self.out_activation = nn.PReLU()

    def forward(self, x):
        main = self.main_branch(x)
        ext  = self.ext_branch(x)
        return self.out_activation(self.batch_norm(torch.cat([main, ext], dim=1)))


# ── DownsamplingBottleneck: MaxPool + AvgPool (MaxAvg) ───────
class DownsamplingBottleneck(nn.Module):
    """Dual-path shortcut: s = MaxPool(X) + AvgPool(X).

    MaxPool giữ indices để dùng cho MaxUnpool trong decoder.
    AvgPool bổ sung thông tin trung bình vùng, cải thiện gradient flow.
    """
    def __init__(self, in_channels, out_channels, dropout_prob=0.01):
        super().__init__()
        internal = out_channels // 4

        self.main_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.main_avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(),
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal, internal, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(),
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.PReLU(),
        )
        self.ext_regul      = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()
        self.out_channels   = out_channels

    def _pad_channels(self, x, target):
        n, ch, h, w = x.size()
        if target > ch:
            pad = torch.zeros(n, target - ch, h, w, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x

    def forward(self, x):
        max_out, max_indices = self.main_maxpool(x)
        avg_out              = self.main_avgpool(x)
        max_out  = self._pad_channels(max_out, self.out_channels)
        avg_out  = self._pad_channels(avg_out, self.out_channels)
        shortcut = max_out + avg_out                   # ← MaxAvg shortcut

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(shortcut + ext), max_indices


# ── RegularBottleneck: DSConv ─────────────────────────────────
class RegularBottleneck(nn.Module):
    """3×3 conv → DSConv (regular, dilated, hoặc asymmetric)."""
    def __init__(self, channels, kernel_size=3, padding=1,
                 dilation=1, asymmetric=False, dropout_prob=0.1):
        super().__init__()
        internal = channels // 4

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(),
        )
        if asymmetric:
            self.ext_conv2 = _dws_conv_asymmetric(
                internal, kernel_size=kernel_size, padding=padding)
        else:
            self.ext_conv2 = _dws_conv(
                internal, kernel_size=kernel_size,
                padding=padding, dilation=dilation)
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.PReLU(),
        )
        self.ext_regul      = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()

    def forward(self, x):
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(x + ext)


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super().__init__()
        internal = in_channels // 4

        self.main_conv  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.main_unpool    = nn.MaxUnpool2d(kernel_size=2)
        self.ext_conv1      = nn.Sequential(
            nn.Conv2d(in_channels, internal, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(),
        )
        self.ext_tconv      = nn.ConvTranspose2d(internal, internal,
                                                 kernel_size=2, stride=2, bias=False)
        self.ext_tconv_bn   = nn.BatchNorm2d(internal)
        self.ext_tconv_act  = nn.PReLU()
        self.ext_conv2      = nn.Sequential(
            nn.Conv2d(internal, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.ext_regul      = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()

    def forward(self, x, max_indices, output_size):
        main = self.main_conv(x)
        main = _max_unpool(self.main_unpool, main, max_indices, output_size)
        ext  = self.ext_conv1(x)
        ext  = self.ext_tconv(ext, output_size=output_size)
        ext  = self.ext_tconv_bn(ext)
        ext  = self.ext_tconv_act(ext)
        ext  = self.ext_conv2(ext)
        ext  = self.ext_regul(ext)
        return self.out_activation(main + ext)


class RegularBottleneckDecoder(nn.Module):
    """Conv 3×3 tiêu chuẩn (không DSConv) theo thiết kế ENet gốc."""
    def __init__(self, channels, dropout_prob=0.1):
        super().__init__()
        internal = channels // 4

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(),
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal, internal, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(),
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.PReLU(),
        )
        self.ext_regul      = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()

    def forward(self, x):
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(x + ext)


# ─────────────────────────────────────────────────────────────
# Model K
# ─────────────────────────────────────────────────────────────

class ENetModel(nn.Module):
    """ENet + DSConv + MaxAvg (Model K).

    Là baseline trực tiếp của FloodENet — khác duy nhất ở chỗ
    chưa có Imp. CA trong decoder.

    Input / Output:
        Input  : [B, 3, 256, 256]
        Output : [B, num_classes, 256, 256]  — raw logits
    """
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        self.initial_block = InitialBlock(in_channels, out_channels=16)

        # Stage 1: 128×128 → 64×64
        self.bottleneck1_0 = DownsamplingBottleneck(16, 64,  dropout_prob=0.01)
        self.bottleneck1_1 = RegularBottleneck(64, dropout_prob=0.01)
        self.bottleneck1_2 = RegularBottleneck(64, dropout_prob=0.01)
        self.bottleneck1_3 = RegularBottleneck(64, dropout_prob=0.01)
        self.bottleneck1_4 = RegularBottleneck(64, dropout_prob=0.01)

        # Stage 2: 64×64 → 32×32
        self.bottleneck2_0 = DownsamplingBottleneck(64, 128, dropout_prob=0.1)
        self.bottleneck2_1 = RegularBottleneck(128, dropout_prob=0.1)
        self.bottleneck2_2 = RegularBottleneck(128, dilation=2,  padding=2,  dropout_prob=0.1)
        self.bottleneck2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck2_4 = RegularBottleneck(128, dilation=4,  padding=4,  dropout_prob=0.1)
        self.bottleneck2_5 = RegularBottleneck(128, dropout_prob=0.1)
        self.bottleneck2_6 = RegularBottleneck(128, dilation=8,  padding=8,  dropout_prob=0.1)
        self.bottleneck2_7 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        # Stage 3: 32×32 (repeat section 2, không downsample)
        self.bottleneck3_0 = RegularBottleneck(128, dropout_prob=0.1)
        self.bottleneck3_1 = RegularBottleneck(128, dilation=2,  padding=2,  dropout_prob=0.1)
        self.bottleneck3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck3_3 = RegularBottleneck(128, dilation=4,  padding=4,  dropout_prob=0.1)
        self.bottleneck3_4 = RegularBottleneck(128, dropout_prob=0.1)
        self.bottleneck3_5 = RegularBottleneck(128, dilation=8,  padding=8,  dropout_prob=0.1)
        self.bottleneck3_6 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        # ── Decoder ──────────────────────────────────────────
        # Stage 4: 32×32 → 64×64
        self.bottleneck4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1)
        self.bottleneck4_1 = RegularBottleneckDecoder(64, dropout_prob=0.1)
        self.bottleneck4_2 = RegularBottleneckDecoder(64, dropout_prob=0.1)

        # Stage 5: 64×64 → 128×128
        self.bottleneck5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1)
        self.bottleneck5_1 = RegularBottleneckDecoder(16, dropout_prob=0.1)

        # Final: 128×128 → 256×256
        self.transposed_conv = nn.ConvTranspose2d(
            16, num_classes, kernel_size=3, stride=2, padding=1,
            output_padding=1, bias=False,
        )

    def forward(self, x):
        x = self.initial_block(x)                                # 16×128×128

        stage1_size = x.size()
        x, max_idx1 = self.bottleneck1_0(x)                      # 64×64×64
        x = self.bottleneck1_1(x); x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x); x = self.bottleneck1_4(x)

        stage2_size = x.size()
        x, max_idx2 = self.bottleneck2_0(x)                      # 128×32×32
        x = self.bottleneck2_1(x);    x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x);    x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x);    x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x);    x = self.bottleneck2_8(x)

        x = self.bottleneck3_0(x);    x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x);    x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x);    x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x);    x = self.bottleneck3_7(x)  # 128×32×32

        x = self.bottleneck4_0(x, max_idx2, output_size=stage2_size)  # 64×64×64
        x = self.bottleneck4_1(x); x = self.bottleneck4_2(x)

        x = self.bottleneck5_0(x, max_idx1, output_size=stage1_size)  # 16×128×128
        x = self.bottleneck5_1(x)

        return self.transposed_conv(x)                           # C×256×256


def build_model(num_classes=1):
    return ENetModel(in_channels=3, num_classes=num_classes)


# ─────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model(num_classes=1).eval()
    dummy = torch.randn(2, 3, 256, 512)
    with torch.no_grad():
        out = model(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Ablation D — ENet + DSConv + MaxAvg  (Model K)]")
    print(f"  Input : {tuple(dummy.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Params: {total:,}")
    print(f"  Changes vs base: +DSConv in RegularBottleneck")
    print(f"                   +MaxAvg in DownsamplingBottleneck")