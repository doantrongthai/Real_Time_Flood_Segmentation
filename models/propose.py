# FloodENet: Model with Improved Coordinate Attention (Imp. CA)
#
# Imp. CA được chèn trước bottleneck4.0 (128→64) và bottleneck5.0 (64→16)
# trong decoder, khớp với kiến trúc trong bài báo (Table 1).
#
# Tên thống nhất với bài báo:
#   - Imp. CA : Improved Coordinate Attention
#   - DWS     : Depthwise Separable convolution (trong encoder)
#   - AP      : Average Pooling thêm vào shortcut của DownsamplingBottleneck

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _max_unpool(unpool_layer, x, indices, output_size):
    """MaxUnpool với tắt deterministic để tránh lỗi CUDA."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    out = unpool_layer(x, indices, output_size=output_size)
    torch.use_deterministic_algorithms(prev)
    return out


def _dws_conv(channels, kernel_size=3, padding=1, dilation=1):
    """Depthwise Separable Convolution (DW + PW) với BN và PReLU.

    Thay thế Conv 3×3 tiêu chuẩn trong encoder bottleneck.
    Giảm chi phí MAC ~9× so với Conv thường (xem Eq.(9)-(11) trong bài).
    """
    return nn.Sequential(
        # Depthwise: Conv kênh-riêng
        nn.Conv2d(channels, channels, kernel_size=kernel_size,
                  padding=padding, dilation=dilation,
                  groups=channels, bias=False),
        nn.BatchNorm2d(channels),
        nn.PReLU(),
        # Pointwise: 1×1 tổng hợp kênh
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.PReLU(),
    )


def _dws_conv_asymmetric(channels, kernel_size=5, padding=2):
    """DWS cho asymmetric factorized filter (5×1 và 1×5).

    Giữ nguyên asymmetric filter từ ENet gốc nhưng dùng DWS.
    """
    return nn.Sequential(
        # Depthwise: 5×1
        nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                  padding=(padding, 0), groups=channels, bias=False),
        nn.BatchNorm2d(channels),
        nn.PReLU(),
        # Depthwise: 1×5
        nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                  padding=(0, padding), groups=channels, bias=False),
        nn.BatchNorm2d(channels),
        nn.PReLU(),
        # Pointwise
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.PReLU(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Improved Coordinate Attention (Imp. CA)
# ─────────────────────────────────────────────────────────────────────────────

class ImprovedCA(nn.Module):
    """Improved Coordinate Attention (Imp. CA).

    Mở rộng CA [Hou et al., CVPR 2021] bằng cách thay thế shared bottleneck
    đơn (1 Conv + h-swish) bằng hai tầng Conv-BN-PReLU liên tiếp trước khi
    tách và chiếu độc lập theo trục H và W.

    Pipeline (khớp với Eq.(1)-(7) trong bài báo):
        Stage 1: Coordinate Information Embedding
            z^h = AvgPool(H,1)(X)   → (B, C, H, 1)   [Eq.(1)]
            z^w = AvgPool(1,W)(X)   → (B, C, 1, W)   [Eq.(2)]
            f0  = cat([z^h, (z^w)^T], dim=2) → (B, C, H+W, 1)  [Eq.(3)]

        Stage 2: Shared Dual-Stage Bottleneck
            f(1) = PReLU(BN(W(1) * f0))   W(1): C→C   [Eq.(4)]
            f    = PReLU(BN(W(2) * f(1))) W(2): C→r   [Eq.(5)]
            r = max(8, C//16)

        Stage 3: Coordinate Attention Generation
            f^h, f^w = Split(f, [H, W], dim=2)         [Eq.(6)]
            g^h = sigmoid(F_h(f^h))  → (B, C, H, 1)   [Eq.(7a)]
            g^w = sigmoid(F_w((f^w)^T)) → (B, C, 1, W)[Eq.(7b)]
            y_c(i,j) = x_c(i,j) * g^h_c(i) * g^w_c(j) [Eq.(8)]

    Args:
        channels  : Số kênh của feature map đầu vào (C).
        reduction : Tỉ lệ nén để tính bottleneck width r = max(8, C//reduction).
                    Mặc định 16, khớp với r = max(8, floor(C/16)) trong bài.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(8, channels // reduction)

        # Stage 1: 1D global average pooling theo hai trục
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # → (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # → (B, C, 1, W)

        # Stage 2: Shared dual-stage bottleneck (dùng PReLU nhất quán với ENet)
        #   Tầng 1: giữ nguyên channel C → C   [Eq.(4), W(1)]
        #   Tầng 2: nén xuống r              [Eq.(5), W(2)]
        self.shared = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, r, kernel_size=1, bias=False),
            nn.BatchNorm2d(r),
            nn.PReLU(),
        )

        # Stage 3: Hai đầu chiếu độc lập [Eq.(7a), (7b)]
        self.fc_h = nn.Conv2d(r, channels, kernel_size=1, bias=False)
        self.fc_w = nn.Conv2d(r, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()

        # Stage 1: Embedding theo trục H và W
        z_h = self.pool_h(x)                              # (B, C, H, 1)
        z_w = self.pool_w(x).permute(0, 1, 3, 2)         # (B, C, W, 1)
        f0  = torch.cat([z_h, z_w], dim=2)               # (B, C, H+W, 1)

        # Stage 2: Shared dual-stage bottleneck
        f = self.shared(f0)                               # (B, r, H+W, 1)

        # Stage 3: Tách + chiếu → attention maps
        f_h, f_w = torch.split(f, [H, W], dim=2)         # (B,r,H,1), (B,r,W,1)
        f_w = f_w.permute(0, 1, 3, 2)                    # (B, r, 1, W)

        g_h = self.sigmoid(self.fc_h(f_h))               # (B, C, H, 1)
        g_w = self.sigmoid(self.fc_w(f_w))               # (B, C, 1, W)

        # Element-wise multiply với broadcasting [Eq.(8)]
        return x * g_h * g_w


# ─────────────────────────────────────────────────────────────────────────────
# ENet Blocks (không thay đổi cấu trúc, chỉ encoder dùng DWS)
# ─────────────────────────────────────────────────────────────────────────────

class InitialBlock(nn.Module):
    """ENet Initial Block: Conv stride-2 // MaxPool → Concat → BN → PReLU."""

    def __init__(self, in_channels: int = 3, out_channels: int = 16):
        super().__init__()
        self.main_branch = nn.Conv2d(
            in_channels, out_channels - in_channels,
            kernel_size=3, stride=2, padding=1, bias=False,
        )
        self.ext_branch   = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batch_norm   = nn.BatchNorm2d(out_channels)
        self.out_activation = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_branch(x)
        ext  = self.ext_branch(x)
        return self.out_activation(self.batch_norm(torch.cat([main, ext], dim=1)))


class DownsamplingBottleneck(nn.Module):
    """ENet Downsampling Bottleneck với dual-path shortcut (AP, Eq.(8) bài báo).

    Shortcut: s = MaxPool(X) + AvgPool(X)  [Eq.(8) trong bài]
    Main branch: Conv 2×2 stride-2 → DW conv → PW 1×1 → Dropout
    """

    def __init__(self, in_channels: int, out_channels: int,
                 dropout_prob: float = 0.01):
        super().__init__()
        internal = out_channels // 4

        # Dual-path shortcut
        self.main_maxpool = nn.MaxPool2d(kernel_size=2, stride=2,
                                         return_indices=True)
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
        self.ext_regul  = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()
        self.out_channels   = out_channels

    def _pad_channels(self, x: torch.Tensor, target: int) -> torch.Tensor:
        n, ch, h, w = x.size()
        if target > ch:
            pad = torch.zeros(n, target - ch, h, w,
                              device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        return x

    def forward(self, x: torch.Tensor):
        max_out, max_idx = self.main_maxpool(x)
        avg_out          = self.main_avgpool(x)
        max_out = self._pad_channels(max_out, self.out_channels)
        avg_out = self._pad_channels(avg_out, self.out_channels)
        shortcut = max_out + avg_out                 # Dual-path [Eq.(8)]

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(shortcut + ext), max_idx


class RegularBottleneck(nn.Module):
    """ENet Regular Bottleneck (encoder).

    3×3 conv → thay bằng DWS (regular, dilated, hoặc asymmetric).
    """

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1,
                 dilation: int = 1, asymmetric: bool = False,
                 dropout_prob: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(x + ext)


class UpsamplingBottleneck(nn.Module):
    """ENet Upsampling Bottleneck (decoder).

    Shortcut: PW 1×1 → MaxUnpool (không dùng DWS ở decoder).
    Main: PW 1×1 → ConvTransposed → PW 1×1 → Dropout.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 dropout_prob: float = 0.1):
        super().__init__()
        internal = in_channels // 4

        self.main_conv  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.main_unpool = nn.MaxUnpool2d(kernel_size=2)

        self.ext_conv1   = nn.Sequential(
            nn.Conv2d(in_channels, internal, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal), nn.PReLU(),
        )
        self.ext_tconv   = nn.ConvTranspose2d(
            internal, internal, kernel_size=2, stride=2, bias=False)
        self.ext_tconv_bn  = nn.BatchNorm2d(internal)
        self.ext_tconv_act = nn.PReLU()
        self.ext_conv2   = nn.Sequential(
            nn.Conv2d(internal, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.ext_regul      = nn.Dropout2d(p=dropout_prob)
        self.out_activation = nn.PReLU()

    def forward(self, x: torch.Tensor, max_indices: torch.Tensor,
                output_size) -> torch.Tensor:
        main = self.main_conv(x)
        main = _max_unpool(self.main_unpool, main, max_indices, output_size)

        ext = self.ext_conv1(x)
        ext = self.ext_tconv(ext, output_size=output_size)
        ext = self.ext_tconv_bn(ext)
        ext = self.ext_tconv_act(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(main + ext)


class RegularBottleneckDecoder(nn.Module):
    """ENet Regular Bottleneck (decoder).

    Dùng Conv 3×3 tiêu chuẩn (không DWS) theo thiết kế ENet gốc.
    """

    def __init__(self, channels: int, dropout_prob: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        return self.out_activation(x + ext)


# ─────────────────────────────────────────────────────────────────────────────
# FloodENet
# ─────────────────────────────────────────────────────────────────────────────

class FloodENet(nn.Module):
    """FloodENet: Lightweight ENet with Improved Coordinate Attention
    for Real-Time Flood Segmentation.

    Kiến trúc (khớp với Table 1 trong bài báo, input 256×256):

    Encoder:
        initial              → 16 × 128 × 128
        bottleneck1.0 (down) → 64 × 64 × 64
        4× bottleneck1.x     → 64 × 64 × 64
        bottleneck2.0 (down) → 128 × 32 × 32
        bottleneck2.1–2.8    → 128 × 32 × 32  (regular/dilated/asymmetric, DWS)
        [Repeat section 2 không có bottleneck2.0]
                             → 128 × 32 × 32

    Decoder:
        Imp. CA              → 128 × 32 × 32
        bottleneck4.0 (up)   → 64 × 64 × 64
        bottleneck4.1–4.2    → 64 × 64 × 64
        Imp. CA              → 64 × 64 × 64
        bottleneck5.0 (up)   → 16 × 128 × 128
        bottleneck5.1        → 16 × 128 × 128
        fullconv (transposed)→ C × 256 × 256
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.initial_block = InitialBlock(in_channels, out_channels=16)

        # Stage 1
        self.bottleneck1_0 = DownsamplingBottleneck(16, 64, dropout_prob=0.01)
        self.bottleneck1_1 = RegularBottleneck(64, dropout_prob=0.01)
        self.bottleneck1_2 = RegularBottleneck(64, dropout_prob=0.01)
        self.bottleneck1_3 = RegularBottleneck(64, dropout_prob=0.01)
        self.bottleneck1_4 = RegularBottleneck(64, dropout_prob=0.01)

        # Stage 2
        self.bottleneck2_0 = DownsamplingBottleneck(64, 128, dropout_prob=0.1)
        self.bottleneck2_1 = RegularBottleneck(128, dropout_prob=0.1)
        self.bottleneck2_2 = RegularBottleneck(128, dilation=2,  padding=2,  dropout_prob=0.1)
        self.bottleneck2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck2_4 = RegularBottleneck(128, dilation=4,  padding=4,  dropout_prob=0.1)
        self.bottleneck2_5 = RegularBottleneck(128, dropout_prob=0.1)
        self.bottleneck2_6 = RegularBottleneck(128, dilation=8,  padding=8,  dropout_prob=0.1)
        self.bottleneck2_7 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        # Stage 3 (repeat section 2, without bottleneck2.0)
        self.bottleneck3_0 = RegularBottleneck(128, dropout_prob=0.1)
        self.bottleneck3_1 = RegularBottleneck(128, dilation=2,  padding=2,  dropout_prob=0.1)
        self.bottleneck3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck3_3 = RegularBottleneck(128, dilation=4,  padding=4,  dropout_prob=0.1)
        self.bottleneck3_4 = RegularBottleneck(128, dropout_prob=0.1)
        self.bottleneck3_5 = RegularBottleneck(128, dilation=8,  padding=8,  dropout_prob=0.1)
        self.bottleneck3_6 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.bottleneck3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Imp. CA trước upsampling 128 → 64
        self.ca_4          = ImprovedCA(128)
        self.bottleneck4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1)
        self.bottleneck4_1 = RegularBottleneckDecoder(64, dropout_prob=0.1)
        self.bottleneck4_2 = RegularBottleneckDecoder(64, dropout_prob=0.1)

        # Imp. CA trước upsampling 64 → 16
        self.ca_5          = ImprovedCA(64)
        self.bottleneck5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1)
        self.bottleneck5_1 = RegularBottleneckDecoder(16, dropout_prob=0.1)

        # Final transposed conv: 16 → num_classes, khôi phục full resolution
        self.transposed_conv = nn.ConvTranspose2d(
            16, num_classes, kernel_size=3, stride=2, padding=1,
            output_padding=1, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size  = x.size()

        # Initial block
        x = self.initial_block(x)                                # 16×128×128

        # Stage 1
        stage1_size = x.size()
        x, max_idx1 = self.bottleneck1_0(x)                      # 64×64×64
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        # Stage 2
        stage2_size = x.size()
        x, max_idx2 = self.bottleneck2_0(x)                      # 128×32×32
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)

        # Stage 3 (repeat section 2)
        x = self.bottleneck3_0(x)
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)                                # 128×32×32

        # Decoder
        x = self.ca_4(x)                                         # Imp. CA
        x = self.bottleneck4_0(x, max_idx2, output_size=stage2_size)  # 64×64×64
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        x = self.ca_5(x)                                         # Imp. CA
        x = self.bottleneck5_0(x, max_idx1, output_size=stage1_size)  # 16×128×128
        x = self.bottleneck5_1(x)

        return self.transposed_conv(x)                           # C×256×256


def build_model(num_classes: int = 1) -> FloodENet:
    """Khởi tạo FloodENet."""
    return FloodENet(in_channels=3, num_classes=num_classes)