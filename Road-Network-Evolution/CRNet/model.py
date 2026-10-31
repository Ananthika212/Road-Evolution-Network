"""
model.py - CRNet definition
===========================
The REAL CRNet architecture, extracted from the official repo
(github.com/CVer-Yang/CRNet) and adapted to run standalone.

Changes made vs. the original repo file (CRNet.py):
  1. Loads ImageNet-pretrained ResNet34 weights via torchvision directly
     (`weights=ResNet34_Weights.IMAGENET1K_V1`) instead of a local
     './networks/resnet34.pth' file, which was not actually included
     in the repo upload.
  2. Removed the hardcoded `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`
     line — Kaggle manages device assignment itself.
  3. Dropped unused classes (Contextblock, Cross_ViewAtt, Cross_MultiViewAtt,
     Mesh_TransformerDecoderLayer, PositionalEncoding, TransformerModel,
     Dblock, Dblock_more_dilate, GlobalContext) that exist in the original
     file but are never used by the actual `CRNet` forward pass — kept only
     what's on the real execution path to keep this file readable.

Naming vs. the paper (for your reference):
  - GLCDM (paper)              -> Cross_ViewAtt1 (this code)
  - SSFRM (paper)               -> Relation_Refine / Relation_Refine2 (this code)

IMPORTANT — input size is NOT flexible:
  Relation_Refine's spatial-attention submodule hardcodes exact patch
  counts (SpatialAttention(4,16), (16,64), (64,256), (256,1024)) that are
  only self-consistent when the input image is exactly 1024x1024 with a
  ResNet34 encoder (giving E1..E4 spatial sizes 512,256,128,64,32 — the
  math is derived in the accompanying notes). Using a different input
  size (e.g. 512x512) WILL cause a shape-mismatch crash inside
  SpatialAttention. Keep IMAGE_SIZE = 1024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

IMAGE_SIZE = 1024  # required — see note above


# ---------------------------------------------------------------------------
# Decoder block (already present in the official repo — no need to build
# a separate decoder, this is it)
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


# ---------------------------------------------------------------------------
# GLCDM equivalent (called Cross_ViewAtt1 in the official code)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class HorizontalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(HorizontalTransformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


class VerticalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(VerticalTransformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(B, W, H, C).permute(0, 3, 2, 1)
        return x


class Cross_ViewAtt1(nn.Module):
    """GLCDM equivalent — global horizontal/vertical context fused with
    local 2x2-patch self-attention via a learned per-patch conv."""

    def __init__(self, in_channels, num_heads):
        super(Cross_ViewAtt1, self).__init__()
        self.in_channels = in_channels
        self.HTransformer = HorizontalTransformer(embed_dim=in_channels, num_heads=num_heads, ff_dim=in_channels * 2, num_layers=1)
        self.VTransformer = VerticalTransformer(embed_dim=in_channels, num_heads=num_heads, ff_dim=in_channels, num_layers=1)
        self.fusion_conv1 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.self_att = TransformerBlock(in_channels, num_heads, in_channels * 2)
        self.fusion_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        ])

    def forward(self, feature_map):
        b = feature_map.size(0)
        Hcontext = self.HTransformer(feature_map)
        Vcontext = self.VTransformer(feature_map)
        fused_out = torch.cat([Hcontext, Vcontext], dim=1)
        global_map = self.fusion_conv1(fused_out)

        n, c, rows, cols = feature_map.size()
        feature_map_1 = feature_map[:, :, :rows // 2, :cols // 2]
        feature_map_2 = feature_map[:, :, :rows // 2, cols // 2:]
        feature_map_3 = feature_map[:, :, rows // 2:, :cols // 2]
        feature_map_4 = feature_map[:, :, rows // 2:, cols // 2:]
        blocks = [feature_map_1, feature_map_2, feature_map_3, feature_map_4]

        attended_outputs = []
        for i in range(4):
            block = blocks[i]
            attended_feature = self.layers[i](block)
            attended_outputs.append(attended_feature)

        stacked_outputs = torch.stack(attended_outputs)
        reshaped_tensor = stacked_outputs.reshape(2, 2, b, self.in_channels, rows // 2, cols // 2)
        permuted_tensor = reshaped_tensor.permute(2, 3, 0, 4, 1, 5)
        output = permuted_tensor.contiguous().reshape(b, self.in_channels, rows, cols)

        output = self.fusion_conv2(global_map + output)
        return output


# ---------------------------------------------------------------------------
# SSFRM equivalent (called Relation_Refine / Relation_Refine2 in the
# official code — Refine2 is only used for the shallowest skip stage)
# ---------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2):
        super(SpatialAttention, self).__init__()
        self.linear = nn.Linear(in_dim1, in_dim2)

    def forward(self, x1, x2):
        x1_transformed = self.linear(x1)
        attention_scores = torch.bmm(x2.transpose(1, 2), x1_transformed)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_x2 = torch.bmm(x2, attention_weights)
        return weighted_x2


class Relation_Refine(nn.Module):
    def __init__(self, in_channels1, in_channels2, stage, groups1=8, groups2=16):
        super(Relation_Refine, self).__init__()
        self.groups1 = groups1
        self.groups2 = groups2
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)

        self.Conv1 = nn.Sequential(
            nn.Conv1d(in_channels1 // groups1, (in_channels1 // 2) // groups1, kernel_size=1),
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv1d(in_channels2 // groups1, in_channels2 // groups1, kernel_size=1),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(in_channels1, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels2, 1, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2)

        self.convbnrelu = nn.Sequential(
            nn.Conv2d(in_channels2, in_channels2, kernel_size=1),
            nn.BatchNorm2d(in_channels2),
            nn.ReLU()
        )
        self.stage = stage
        if self.stage == 4:
            self.sa = SpatialAttention(4, 16)
        elif self.stage == 3:
            self.sa = SpatialAttention(16, 64)
        elif self.stage == 2:
            self.sa = SpatialAttention(64, 256)
        elif self.stage == 1:
            self.sa = SpatialAttention(256, 1024)

    def forward(self, x1, x2):
        # x1: deeper/decoder-side feature (fewer spatial, more channels)
        # x2: shallow skip-connection feature (more spatial, fewer channels)
        B, C1, H1, W1 = x1.size()
        B, C2, H2, W2 = x2.size()

        # ---- semantic (channel) refinement ----
        sem_refine_x2_list = []
        for i in range(self.groups1):
            x1_group = x1[:, i * (C1 // self.groups1):(i + 1) * (C1 // self.groups1), :, :]
            x2_group = x2[:, i * (C2 // self.groups1):(i + 1) * (C2 // self.groups1), :, :]

            gap1 = self.gap1(x1_group).view(x1_group.size(0), x1_group.size(1), -1)
            gap2 = self.gap2(x2_group).view(x2_group.size(0), x2_group.size(1), -1)

            mlp1 = self.Conv1(gap1)
            mlp2 = self.Conv2(gap2)
            sem_matrix = torch.bmm(mlp1, mlp2.transpose(1, 2))

            channel_weight = torch.softmax(torch.bmm(sem_matrix, mlp2), dim=1).view(
                x2_group.size(0), x2_group.size(1), 1, 1)
            sem_refine_x2 = x2_group * channel_weight
            sem_refine_x2_list.append(sem_refine_x2)

        sem_refine_x2 = torch.cat(sem_refine_x2_list, dim=1)

        # ---- spatial refinement ----
        spa_refine_x2_list = []
        split_size_h1 = H1 // self.groups2
        split_size_w1 = W1 // self.groups2
        split_size_h2 = H2 // self.groups2
        split_size_w2 = W2 // self.groups2

        for i in range(self.groups2):
            for j in range(self.groups2):
                x1_group = x1[:, :, i * split_size_h1:(i + 1) * split_size_h1, j * split_size_w1:(j + 1) * split_size_w1]
                x2_group = x2[:, :, i * split_size_h2:(i + 1) * split_size_h2, j * split_size_w2:(j + 1) * split_size_w2]

                pos1 = self.conv1(x1_group).view(x1_group.size(0), 1, -1)
                pos2 = self.conv2(x2_group).view(x2_group.size(0), 1, -1)
                spa_refine_x2 = self.sa(pos1, pos2)
                spa_refine_x2_list.append(spa_refine_x2)

        spa_refine_x2 = torch.cat(spa_refine_x2_list, dim=2).view(B, 1, H2, W2)

        fuse_feature = self.convbnrelu(sem_refine_x2 + spa_refine_x2)
        return fuse_feature


class Relation_Refine2(nn.Module):
    """Same as Relation_Refine but the semantic-branch reduction conv
    doesn't halve the channel count (used only for the shallowest,
    equal-channel skip stage)."""

    def __init__(self, in_channels1, in_channels2, stage, groups1=8, groups2=16):
        super(Relation_Refine2, self).__init__()
        self.groups1 = groups1
        self.groups2 = groups2
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)

        self.Conv1 = nn.Sequential(
            nn.Conv1d(in_channels1 // groups1, in_channels1 // groups1, kernel_size=1),
            nn.ReLU()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv1d(in_channels2 // groups1, in_channels2 // groups1, kernel_size=1),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(in_channels1, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels2, 1, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2)

        self.convbnrelu = nn.Sequential(
            nn.Conv2d(in_channels2, in_channels2, kernel_size=1),
            nn.BatchNorm2d(in_channels2),
            nn.ReLU()
        )
        self.stage = stage
        if self.stage == 4:
            self.sa = SpatialAttention(4, 16)
        elif self.stage == 3:
            self.sa = SpatialAttention(16, 64)
        elif self.stage == 2:
            self.sa = SpatialAttention(64, 256)
        elif self.stage == 1:
            self.sa = SpatialAttention(256, 1024)

    def forward(self, x1, x2):
        B, C1, H1, W1 = x1.size()
        B, C2, H2, W2 = x2.size()

        sem_refine_x2_list = []
        for i in range(self.groups1):
            x1_group = x1[:, i * (C1 // self.groups1):(i + 1) * (C1 // self.groups1), :, :]
            x2_group = x2[:, i * (C2 // self.groups1):(i + 1) * (C2 // self.groups1), :, :]

            gap1 = self.gap1(x1_group).view(x1_group.size(0), x1_group.size(1), -1)
            gap2 = self.gap2(x2_group).view(x2_group.size(0), x2_group.size(1), -1)

            mlp1 = self.Conv1(gap1)
            mlp2 = self.Conv2(gap2)
            sem_matrix = torch.bmm(mlp1, mlp2.transpose(1, 2))

            channel_weight = torch.softmax(torch.bmm(sem_matrix, mlp2), dim=1).view(
                x2_group.size(0), x2_group.size(1), 1, 1)
            sem_refine_x2 = x2_group * channel_weight
            sem_refine_x2_list.append(sem_refine_x2)

        sem_refine_x2 = torch.cat(sem_refine_x2_list, dim=1)

        spa_refine_x2_list = []
        split_size_h1 = H1 // self.groups2
        split_size_w1 = W1 // self.groups2
        split_size_h2 = H2 // self.groups2
        split_size_w2 = W2 // self.groups2

        for i in range(self.groups2):
            for j in range(self.groups2):
                x1_group = x1[:, :, i * split_size_h1:(i + 1) * split_size_h1, j * split_size_w1:(j + 1) * split_size_w1]
                x2_group = x2[:, :, i * split_size_h2:(i + 1) * split_size_h2, j * split_size_w2:(j + 1) * split_size_w2]

                pos1 = self.conv1(x1_group).view(x1_group.size(0), 1, -1)
                pos2 = self.conv2(x2_group).view(x2_group.size(0), 1, -1)
                spa_refine_x2 = self.sa(pos1, pos2)
                spa_refine_x2_list.append(spa_refine_x2)

        spa_refine_x2 = torch.cat(spa_refine_x2_list, dim=2).view(B, 1, H2, W2)

        fuse_feature = self.convbnrelu(sem_refine_x2 + spa_refine_x2)
        return fuse_feature


# ---------------------------------------------------------------------------
# Full CRNet
# ---------------------------------------------------------------------------
class CRNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super(CRNet, self).__init__()

        filters = [64, 128, 256, 512]

        try:
            from torchvision.models import ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
        except ImportError:
            # older torchvision fallback
            resnet = models.resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.context = Cross_ViewAtt1(in_channels=512, num_heads=8)   # GLCDM
        self.RR4 = Relation_Refine(512, 256, 4)                         # SSFRM stage 4
        self.RR3 = Relation_Refine(256, 128, 3)                         # SSFRM stage 3
        self.RR2 = Relation_Refine(128, 64, 2)                          # SSFRM stage 2
        self.RR1 = Relation_Refine2(64, 64, 1)                          # SSFRM stage 1

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image):
        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)          # (B, 64, H/2, W/2)  -- used later as RR1's shallow feature
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)          # (B, 64,  H/4,  W/4)
        e2 = self.encoder2(e1)          # (B, 128, H/8,  W/8)
        e3 = self.encoder3(e2)          # (B, 256, H/16, W/16)
        e4 = self.encoder4(e3)          # (B, 512, H/32, W/32)
        e4 = self.context(e4)           # GLCDM

        # Decoder + SSFRM skip connections
        d4 = self.decoder4(e4) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3, e1)
        d1 = self.decoder1(d2) + self.RR1(d2, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


if __name__ == "__main__":
    # Shape sanity check — run this to confirm the model builds and a
    # forward pass produces the expected output shape, BEFORE using it
    # in the full training script.
    torch.manual_seed(0)
    model = CRNet(num_classes=1, pretrained=False)  # pretrained=False here just to avoid a network call in this quick check
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out = model(dummy)
    print("Input shape :", dummy.shape)
    print("Output shape:", out.shape)   # expect (1, 1, 1024, 1024)
