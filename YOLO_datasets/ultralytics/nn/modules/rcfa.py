import torch
import torch.nn as nn
import torch.nn.functional as F


class RCFA(nn.Module):
    """
    RCFA v2: Orientation-aware Feature Alignment
    - 用多方向深度卷积 + 方向权重做“旋转一致性特征对齐”
    - 不再做几何旋转采样（没有 affine_grid / grid_sample，训练更稳）
    - 自带残差 + 通道门控，可退化成近似恒等映射
    """

    def __init__(self, channels, num_dirs: int = 4, reduction: int = 4):
        """
        Args:
            channels: 输入通道数（如 128 / 256，或通道列表中的第一个）
            num_dirs: 方向数（等价于多少个方向卷积核，默认 4 个）
            reduction: 通道注意力的压缩比
        """
        super().__init__()
        c = channels if isinstance(channels, int) else channels[0]
        self.c = c
        self.num_dirs = num_dirs

        # 1️⃣ 多方向深度可分离卷积：为每个通道生成 num_dirs 个方向响应
        # 输出通道 = c * num_dirs，groups=c 代表按通道独立卷积
        self.dir_conv = nn.Conv2d(
            c, c * num_dirs, kernel_size=3, padding=1,
            groups=c, bias=False
        )
        self.dir_bn = nn.BatchNorm2d(c * num_dirs)
        self.dir_act = nn.ReLU(inplace=True)

        # 2️⃣ 方向权重预测：从原始特征中预测每个方向的权重
        self.dir_weight_branch = nn.Sequential(
            nn.Conv2d(c, c // reduction, 1, bias=False),
            nn.BatchNorm2d(c // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // reduction, num_dirs, 1)  # [B, num_dirs, H, W]
        )

        # 3️⃣ 通道门控（类似 SE）：控制增强特征在每个通道上的贡献
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // reduction, c, 1),
            nn.Sigmoid()
        )

        # 4️⃣ 线性投影（用于轻微重整方向增强特征）
        self.proj = nn.Conv2d(c, c, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c)

        # 5️⃣ 全局门控（可选）：控制 RCFA 整体强度，初始化为“很小”
        self.global_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

    def forward(self, x):
        """
        x: [B, C, H, W]
        输出与输入同形状
        """
        B, C, H, W = x.shape

        # ---- 多方向响应 ----
        # dir_feat_raw: [B, C*num_dirs, H, W]
        dir_feat_raw = self.dir_act(self.dir_bn(self.dir_conv(x)))

        # reshape 为 [B, num_dirs, C, H, W]
        dir_feat = dir_feat_raw.view(B, self.num_dirs, C, H, W)

        # ---- 方向权重 ----
        # logits: [B, num_dirs, H, W]
        dir_logits = self.dir_weight_branch(x)
        # weights: [B, num_dirs, 1, H, W]，在方向维度 softmax
        dir_weights = torch.softmax(dir_logits, dim=1).unsqueeze(2)

        # ---- 按方向加权求和 → 方向对齐后的特征 ----
        # [B, C, H, W]
        feat_oriented = (dir_feat * dir_weights).sum(dim=1)

        # 线性投影 + BN（保持数值稳定）
        feat_oriented = self.proj_bn(self.proj(feat_oriented))

        # ---- 通道门控 ----
        ch_gate = self.channel_gate(x)  # [B, C, 1, 1]

        # ---- 全局门控（标量） ----
        g = torch.sigmoid(self.global_gate)  # (0, 1) 之间

        # 最终输出：残差增强
        # out = x + g * ch_gate * (feat_oriented - x)
        out = x + g * ch_gate * (feat_oriented - x)

        return out