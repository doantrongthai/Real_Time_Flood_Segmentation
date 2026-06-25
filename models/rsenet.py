import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────
# Feature Enhancement Block (FEB)
# Paper text (Section 2.2): Conv2D(1×1) → PReLU → Conv2D(3×3) → PReLU → Conv2D(1×1) → PReLU → Dropout(0.01)
# NOTE: Figure 3 shows three 1×1 convolutions, but the body text explicitly states
#       "The second convolution layer captures spatial features with 3×3 filters",
#       which matches the standard bottleneck / inverted-bottleneck pattern (1×1→3×3→1×1).
#       Body text takes precedence over the figure label.
# ─────────────────────────────────────────────
class FEBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1),
            nn.PReLU(in_c),
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),  # 3×3 per paper body text
            nn.PReLU(in_c),
            nn.Conv2d(in_c, in_c, kernel_size=1),
            nn.PReLU(in_c),
            nn.Dropout2d(p=0.01),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# SE Block (standard Squeeze-and-Excitation)
# Used inside R-SE Block
# ─────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # Squeeze
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # returns attention weights: (B, C, 1, 1)
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return w


# ─────────────────────────────────────────────
# Residual Squeeze-and-Excitation Block (R-SE)
# Figure 2 (bottom → top):
#   input
#   → CONV → ReLU → CONV
#   → Concat(with input)
#   → 1×1 CONV
#   → SE Block  (produces channel attention weights)
#   → ⊗ (element-wise mul with 1×1 CONV output)
#   → ⊕ (residual add with original input)
#   → output
# ─────────────────────────────────────────────
class RSEBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        # Two convolutions with ReLU between them
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, bias=False)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, bias=False)

        # 1×1 conv after concat (in_c + in_c → in_c)
        self.pw = nn.Conv2d(in_c * 2, in_c, kernel_size=1, bias=False)

        # SE block applied on the projected features
        self.se = SEBlock(in_c)

    def forward(self, x):
        identity = x                               # for residual addition

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)                      # (B, C, H, W)

        # Concat along channel dim with original input, then project
        out = torch.cat([out, x], dim=1)           # (B, 2C, H, W)
        out = self.pw(out)                         # (B, C, H, W)

        # Channel attention via SE
        attn = self.se(out)                        # (B, C, 1, 1)
        out  = out * attn                          # ⊗ element-wise mul

        out  = out + identity                      # ⊕ residual add
        return out


# ─────────────────────────────────────────────
# RSENet  (Figure 1)
#
# Backbone: ResNet-50 (first 4 layer groups)
# Channel widths follow ResNet-50 defaults:
#   layer1 → 256c,  layer2 → 512c,
#   layer3 → 1024c, layer4 → 2048c
# But the paper shows identical-width boxes labeled
# "×c" at all scales, so we reduce with 1×1 adapters.
#
# For input 256×256 the spatial sizes are:
#   after stem (conv+bn+relu+maxpool): 64×64
#   after layer1: 64×64   (c=256  →  adapt to base_c)
#   after layer2: 32×32   (c=512  →  adapt to base_c)
#   after layer3: 16×16   (c=1024 →  adapt to base_c)
#   after layer4:  8×8    (c=2048 →  adapt to base_c)
#
# We keep base_c = 128 for a lightweight but expressive model.
#
# Processing columns (Figure 1, left-to-right):
#   Col-0  (residual block outputs, 4 scales)
#   Col-1  after R-SE on scale-0; FE on all 4 scales
#   Col-2  after R-SE on scale-1; FE on scales 1-3 (scale-0 not shown)
#   Col-3  after R-SE on scale-2; FE on scales 2-3 (scale-0,1 not shown)
#
# NOTE on R-SE count:
#   Paper body text (Section 2.1) states R-SE blocks are placed "after the RB-3 and
#   RB-4 stages" (2 blocks). Figure 1 however shows 3 R-SE blocks (at scales 0, 1, 2).
#   This implementation follows Figure 1 (3 R-SE blocks) as the figure is more
#   architecturally specific than the prose summary.
#
# Final: 4× upsample from s0 (64×64 @ 256-input) → 256×256 (full resolution).
# ─────────────────────────────────────────────
class RSENetModel(nn.Module):
    """
    Flood segmentation model based on:
    "A Flood Segmentation Model Enhanced by Residual Squeeze-and-Excitation (R-SE) Blocks"
    (Güçlü et al., APJESS 2026).

    Input:  (B, 3, 256, 256)
    Output: (B, num_classes, 256, 256)  — raw logits (no sigmoid)
    """

    def __init__(self, num_classes=1, base_c=128):
        super().__init__()

        # ── Backbone (ResNet-50, pretrained weights optional) ──────────────
        backbone = models.resnet50(weights=None)

        # Stem: conv1 + bn1 + relu + maxpool  →  (B, 64, H/4, W/4)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        # 4 residual layer groups
        self.rb1 = backbone.layer1   # out: 256c,  H/4
        self.rb2 = backbone.layer2   # out: 512c,  H/8
        self.rb3 = backbone.layer3   # out: 1024c, H/16
        self.rb4 = backbone.layer4   # out: 2048c, H/32

        # Channel adapters: squeeze ResNet channels → base_c
        self.adapt1 = nn.Conv2d(256,  base_c, kernel_size=1, bias=False)
        self.adapt2 = nn.Conv2d(512,  base_c, kernel_size=1, bias=False)
        self.adapt3 = nn.Conv2d(1024, base_c, kernel_size=1, bias=False)
        self.adapt4 = nn.Conv2d(2048, base_c, kernel_size=1, bias=False)

        c = base_c

        # ── Column 1: R-SE on scale-0; FE on all 4 scales ──────────────────
        self.rse_col1_s0 = RSEBlock(c)      # applied to scale-0 (64×64 @ 256-input)
        self.feb_col1_s0 = FEBlock(c)
        self.feb_col1_s1 = FEBlock(c)
        self.feb_col1_s2 = FEBlock(c)
        self.feb_col1_s3 = FEBlock(c)

        # ── Column 2: R-SE on scale-1; FE on scales 1–3 ────────────────────
        self.rse_col2_s1 = RSEBlock(c)      # applied to scale-1 (32×32 @ 256-input)
        self.feb_col2_s1 = FEBlock(c)
        self.feb_col2_s2 = FEBlock(c)
        self.feb_col2_s3 = FEBlock(c)

        # ── Column 3: R-SE on scale-2; FE on scales 2–3 ────────────────────
        self.rse_col3_s2 = RSEBlock(c)      # applied to scale-2 (16×16 @ 256-input)
        self.feb_col3_s2 = FEBlock(c)
        self.feb_col3_s3 = FEBlock(c)

        # ── Decoder: progressive 2× upsampling from scale-3 (8×8) → 256 ───
        # Each step: upsample + fuse (add) with higher-res feature + conv
        # step: 8 → 16 → 32 → 64 → 128 → 256  (5 × 2×)
        # But we only have meaningful skip connections at 3 scales
        # (col3_s2, col2_s1, col1_s0), then bilinear × 4 to reach 256.
        self.up_s3_to_s2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.PReLU(c),
        )
        self.fuse_s2 = nn.Conv2d(c * 2, c, kernel_size=1, bias=False)

        self.up_s2_to_s1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.PReLU(c),
        )
        self.fuse_s1 = nn.Conv2d(c * 2, c, kernel_size=1, bias=False)

        self.up_s1_to_s0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.PReLU(c),
        )
        self.fuse_s0 = nn.Conv2d(c * 2, c, kernel_size=1, bias=False)

        # Final 4× upsample (mirrors paper's "4×4 Upsample")
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.PReLU(c // 2),
        )

        # Output head (Sigmoid applied externally during loss/inference)
        self.head = nn.Conv2d(c // 2, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # ── Backbone ────────────────────────────────────────────────────────
        x  = self.stem(x)             # (B, 64, H/4, W/4)
        f1 = self.rb1(x)              # (B, 256,  H/4,  W/4)   scale-0
        f2 = self.rb2(f1)             # (B, 512,  H/8,  W/8)   scale-1
        f3 = self.rb3(f2)             # (B, 1024, H/16, W/16)  scale-2
        f4 = self.rb4(f3)             # (B, 2048, H/32, W/32)  scale-3

        # Channel adapters → base_c at every scale
        s0 = self.adapt1(f1)          # (B, c, H/4,  W/4)
        s1 = self.adapt2(f2)          # (B, c, H/8,  W/8)
        s2 = self.adapt3(f3)          # (B, c, H/16, W/16)
        s3 = self.adapt4(f4)          # (B, c, H/32, W/32)

        # ── Column 1 ─────────────────────────────────────────────────────────
        # R-SE on scale-0, then FE on all 4 scales
        c1_s0 = self.feb_col1_s0(self.rse_col1_s0(s0))
        c1_s1 = self.feb_col1_s1(s1)
        c1_s2 = self.feb_col1_s2(s2)
        c1_s3 = self.feb_col1_s3(s3)

        # ── Column 2 ─────────────────────────────────────────────────────────
        # R-SE on scale-1 (from col-1), then FE on scales 1–3
        c2_s1 = self.feb_col2_s1(self.rse_col2_s1(c1_s1))
        c2_s2 = self.feb_col2_s2(c1_s2)
        c2_s3 = self.feb_col2_s3(c1_s3)

        # ── Column 3 ─────────────────────────────────────────────────────────
        # R-SE on scale-2 (from col-2), then FE on scales 2–3
        c3_s2 = self.feb_col3_s2(self.rse_col3_s2(c2_s2))
        c3_s3 = self.feb_col3_s3(c2_s3)

        # ── Decoder (bottom-up, fusing multi-scale features) ─────────────────
        # s3 (deepest) → up → fuse with c3_s2 → up → fuse with c2_s1
        #             → up → fuse with c1_s0 → 4× up → head

        d = self.up_s3_to_s2(c3_s3)                     # H/16
        d = self.fuse_s2(torch.cat([d, c3_s2], dim=1))  # fuse col3-s2

        d = self.up_s2_to_s1(d)                          # H/8
        d = self.fuse_s1(torch.cat([d, c2_s1], dim=1))  # fuse col2-s1

        d = self.up_s1_to_s0(d)                          # H/4
        d = self.fuse_s0(torch.cat([d, c1_s0], dim=1))  # fuse col1-s0

        d = self.final_up(d)                             # H (full resolution)
        out = self.head(d)                               # (B, num_classes, H, W)
        return out


def build_model(num_classes=1):
    return RSENetModel(num_classes=num_classes)
