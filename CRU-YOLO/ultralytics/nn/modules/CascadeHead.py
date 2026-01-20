import math

import torch
from torch import nn
from ultralytics.nn.modules import Detect, Conv
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors

class RABR(nn.Module):
    """
    Rotation-Aware Bounding Box Regression (RABR)
    - 轻量化方向感知特征模块
    - 放在 OBB head 内增强方向建模能力
    """
    def __init__(self, in_channels, ratio=0.25):
        super().__init__()
        mid_c = max(int(in_channels * ratio), 16)
        self.conv_angle = nn.Sequential(
            Conv(in_channels, mid_c, 3),
            Conv(mid_c, in_channels, 3)
        )
        # 可学习融合系数
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat_angle = self.conv_angle(x)
        w = self.sigmoid(self.alpha)
        return w * x + (1 - w) * feat_angle

class RABR_C(nn.Module):
    """
    Rotation-Aware Bounding Box Regression (RABR-C)
    - 通道注意力增强版：在RABR-S基础上加入SE-like注意力
    - 先做方向卷积增强，再通过通道注意力自适应加权
    """
    def __init__(self, in_channels, ratio=0.25, reduction=16):
        super().__init__()
        mid_c = max(int(in_channels * ratio), 16)

        # 🔹 方向卷积增强分支（同RABR-S）
        self.conv_angle = nn.Sequential(
            Conv(in_channels, mid_c, 3),
            Conv(mid_c, in_channels, 3)
        )

        # 🔹 通道注意力分支（SE结构）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

        # 🔹 可学习融合系数
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Step1: 方向卷积增强
        feat_angle = self.conv_angle(x)

        # Step2: 通道注意力
        b, c, _, _ = x.size()
        y = self.global_pool(feat_angle).view(b, c)
        attn = self.fc(y).view(b, c, 1, 1)  # 通道权重 [B,C,1,1]

        # Step3: 通道加权融合
        feat_attn = feat_angle * attn
        w = self.sigmoid(self.alpha)
        return w * x + (1 - w) * feat_attn

class RABR_M(nn.Module):
    """
    Multi-scale Rotation-Aware Feature Fusion (RABR-M)
    - 对每个尺度：先做RABR_S方向增强
    - 与上一级(更粗)做上采样融合、与下一级(更细)做下采样融合
    - concat后用1×1Conv投影回原通道
    """
    def __init__(self, channels: tuple):
        super().__init__()
        self.nl = len(channels)
        self.rabr = nn.ModuleList(RABR(c) for c in channels)

        self.up_map = nn.ModuleList()
        self.down_map = nn.ModuleList()
        self.merge = nn.ModuleList()

        for i in range(self.nl):
            # 上采样映射：from i+1 → i
            if i < self.nl - 1:
                self.up_map.append(Conv(channels[i + 1], channels[i], 1, 1))
            else:
                self.up_map.append(nn.Identity())

            # 下采样映射：from i-1 → i
            if i > 0:
                self.down_map.append(Conv(channels[i - 1], channels[i], 3, 2))
            else:
                self.down_map.append(nn.Identity())

            # 融合后1×1投影
            in_cat = channels[i]
            if i < self.nl - 1: in_cat += channels[i]
            if i > 0: in_cat += channels[i]
            self.merge.append(Conv(in_cat, channels[i], 1, 1))

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, feats):
        assert len(feats) == self.nl
        base = [self.rabr[i](feats[i]) for i in range(self.nl)]
        out = []

        for i in range(self.nl):
            parts = [base[i]]

            # 上采样来自更粗层
            if i < self.nl - 1:
                up = self.up_map[i](base[i + 1])
                up = self.upsample(up)
                if up.shape[-2:] != base[i].shape[-2:]:
                    up = nn.functional.interpolate(up, size=base[i].shape[-2:], mode="nearest")
                parts.append(up)

            # 下采样来自更细层
            if i > 0:
                down = self.down_map[i](base[i - 1])
                if down.shape[-2:] != base[i].shape[-2:]:
                    down = nn.functional.interpolate(down, size=base[i].shape[-2:], mode="nearest")
                parts.append(down)

            fused = torch.cat(parts, 1)
            fused = self.merge[i](fused)
            out.append(fused)
        return out

class ChannelAttention(nn.Module):
    """SE结构：全局平均池化 + FC + Sigmoid"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        w = self.fc(y).view(b, c, 1, 1)
        return x * w

class RABR_MC(nn.Module):
    """
    Multi-scale + Channel Attention Rotation-Aware Fusion (RABR-MC)
    - 融合多尺度方向一致性和通道注意力
    - 对每个尺度：RABR增强 → 上下尺度融合 → 通道加权 → 投影
    """
    def __init__(self, channels: tuple, reduction=16):
        super().__init__()
        self.nl = len(channels)
        self.rabr = nn.ModuleList(RABR(c) for c in channels)
        self.ca = nn.ModuleList(ChannelAttention(c, reduction) for c in channels)

        self.up_map = nn.ModuleList()
        self.down_map = nn.ModuleList()
        self.merge = nn.ModuleList()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        for i in range(self.nl):
            if i < self.nl - 1:
                self.up_map.append(Conv(channels[i + 1], channels[i], 1, 1))
            else:
                self.up_map.append(nn.Identity())

            if i > 0:
                self.down_map.append(Conv(channels[i - 1], channels[i], 3, 2))
            else:
                self.down_map.append(nn.Identity())

            in_cat = channels[i]
            if i < self.nl - 1:
                in_cat += channels[i]
            if i > 0:
                in_cat += channels[i]
            self.merge.append(Conv(in_cat, channels[i], 1, 1))

    def forward(self, feats):
        base = [self.rabr[i](feats[i]) for i in range(self.nl)]
        out = []

        for i in range(self.nl):
            parts = [base[i]]

            # 上采样来自更粗层
            if i < self.nl - 1:
                up = self.up_map[i](base[i + 1])
                up = self.upsample(up)
                if up.shape[-2:] != base[i].shape[-2:]:
                    up = nn.functional.interpolate(up, size=base[i].shape[-2:], mode="nearest")
                parts.append(up)

            # 下采样来自更细层
            if i > 0:
                down = self.down_map[i](base[i - 1])
                if down.shape[-2:] != base[i].shape[-2:]:
                    down = nn.functional.interpolate(down, size=base[i].shape[-2:], mode="nearest")
                parts.append(down)

            fused = torch.cat(parts, dim=1)
            fused = self.merge[i](fused)

            # 通道注意力再增强
            fused = self.ca[i](fused)
            out.append(fused)
        return out



class CrossStageAttention(nn.Module):
    """Cross-Stage Attention (CSA)
    x:        当前阶段特征 [B, C, H, W]
    prev_feat:上一阶段同尺度特征 [B, C, H, W]
    """
    def __init__(self, channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or max(1, channels // 2)
        self.query_conv = nn.Conv2d(channels, inter_channels, 1)
        self.key_conv   = nn.Conv2d(channels, inter_channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, prev_feat):
        if prev_feat is None:
            return x

        # 强约束：CSA 两端通道必须一致（由上层结构保证）
        if prev_feat.shape[1] != x.shape[1]:
            raise RuntimeError(f"CSA channel mismatch: x={tuple(x.shape)}, prev={tuple(prev_feat.shape)}")

        q = self.query_conv(x)
        k = self.key_conv(prev_feat)
        v = self.value_conv(prev_feat)

        attn = torch.sigmoid(torch.mean(q * k, dim=1, keepdim=True))  # [B,1,H,W]
        return x + self.gamma.to(x.dtype) * (attn * v)

class CSAOrIdentity(nn.Module):
    """统一 CSA/Identity 接口：避免 ModuleList 里放 None，也避免 Identity 不支持两个入参"""
    def __init__(self, channels: int, enabled: bool):
        super().__init__()
        self.enabled = bool(enabled)
        self.block = CrossStageAttention(channels) if self.enabled else nn.Identity()

    def forward(self, x, prev_feat=None):
        if (not self.enabled) or (prev_feat is None):
            return x
        return self.block(x, prev_feat)


# =========================
# RABR factory + wrapper
# =========================
class _PerLevelWrapper(nn.Module):
    """把单输入模块扩展到多尺度 list[feat] 的 wrapper"""
    def __init__(self, channels, ctor):
        super().__init__()
        self.blocks = nn.ModuleList([ctor(c) for c in channels])

    def forward(self, feats):
        assert isinstance(feats, (list, tuple)), f"feats must be list/tuple, got {type(feats)}"
        assert len(feats) == len(self.blocks), f"len(feats)={len(feats)} != nl={len(self.blocks)}"
        return [blk(f) for blk, f in zip(self.blocks, feats)]


def build_rabr_block(channels, mode: str):
    """
    mode:
      - "none": 不用
      - "s":   RABR  (per-level)
      - "c":   RABR_C(per-level)
      - "m":   RABR_M
      - "mc":  RABR_MC
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return nn.Identity()
    if mode == "s":
        return _PerLevelWrapper(channels, lambda c: RABR(c))
    if mode == "c":
        return _PerLevelWrapper(channels, lambda c: RABR_C(c))
    if mode == "m":
        return RABR_M(channels)
    if mode == "mc":
        return RABR_MC(channels)
    raise ValueError(f"Unknown rabr mode: {mode}")


# =========================
# OBB Cascade Head
# =========================
class OBB_CascadeHead(Detect):
    """
    Cascade ROI-based OBB Head (CSA + optional RABR)
    - 训练时可指定启用/禁用：RABR / CSA / 级联阶段数
    - ✅ 修复 Ultralytics build-time stride 推断：dummy forward 只返回 det
    """
    def __init__(
        self,
        nc=80,
        ne=1,
        cascade_stages=2,
        use_csa=True,
        rabr_mode="mc",
        return_all_stages=False,
        debug=False,
        ch=(),  # ✅ 放最后！parse_model 会 append 到最后
    ):
        super().__init__(nc, ch)

        self.ne = int(ne)
        self.nl = len(ch)
        self.cascade_stages = int(cascade_stages)
        self.use_csa = bool(use_csa)
        self.return_all_stages = bool(return_all_stages)
        self.debug = bool(debug)

        # ✅ build-time stride inference gate
        # Ultralytics 会在构建阶段跑一次 dummy forward 来推 stride，
        # 这时 head 必须返回 “可遍历的 tensor 列表/张量”，不能返回 (det, angle) 这种嵌套结构。
        self._stride_infer = True

        channels = tuple(ch)
        c4 = max(min(channels) // 4, 16, self.ne)

        # 1) Optional RABR
        self.rabr = build_rabr_block(channels, rabr_mode)

        # 2) Cascade branches
        self.trunks = nn.ModuleList()
        self.angle_preds = nn.ModuleList()
        self.angle_embeds = nn.ModuleList()

        for s in range(self.cascade_stages):
            trunk_s = nn.ModuleList()
            pred_s  = nn.ModuleList()
            emb_s   = nn.ModuleList()
            for xch in channels:
                trunk_s.append(nn.Sequential(
                    Conv(xch, c4, 3),
                    Conv(c4, c4, 3),
                ))
                pred_s.append(nn.Conv2d(c4, self.ne, 1))
                emb_s.append(nn.Conv2d(self.ne, xch, 1))
            self.trunks.append(trunk_s)
            self.angle_preds.append(pred_s)
            self.angle_embeds.append(emb_s)

        # 3) CSA blocks
        self.csa_blocks = nn.ModuleList()
        for s in range(self.cascade_stages):
            csa_s = nn.ModuleList()
            for xch in channels:
                csa_s.append(CSAOrIdentity(xch, enabled=(self.use_csa and s > 0)))
            self.csa_blocks.append(csa_s)

        print(f"[INFO] OBB_CascadeHead: stages={self.cascade_stages}, CSA={self.use_csa}, RABR={rabr_mode}")

    def forward(self, x):
        """
        x: list/tuple of multi-level features
        """
        bs = x[0].shape[0]

        # Step1: optional RABR
        feats = self.rabr(x)

        # Step2: cascade angle refinement (+ optional CSA)
        prev_embed = None
        all_angles = []

        for s in range(self.cascade_stages):
            angle_list = []
            curr_embed = []

            for i in range(self.nl):
                xi = feats[i]

                # CSA
                xi = self.csa_blocks[s][i](xi, None if prev_embed is None else prev_embed[i])

                trunk = self.trunks[s][i](xi)                 # [B, c4, H, W]
                angle_logits = self.angle_preds[s][i](trunk)  # [B, ne, H, W]

                angle_list.append(angle_logits.view(bs, self.ne, -1))

                emb = torch.tanh(self.angle_embeds[s][i](angle_logits))
                curr_embed.append(emb)

            angle_stage = torch.cat(angle_list, dim=2)
            angle_stage = (angle_stage.sigmoid() - 0.25) * math.pi
            all_angles.append(angle_stage)

            if self.debug:
                mn, mx = angle_stage.min().item(), angle_stage.max().item()
                print(f"[Stage {s}] angle range=({mn:.3f}, {mx:.3f})")

            prev_embed = curr_embed

        final_angle = all_angles[-1]
        if not self.training:
            self.angle = final_angle

        # Step3: YOLO detect output
        det = super().forward(feats)

        # =========================================================
        # ✅ Fix for Ultralytics build-time stride inference
        # During model building, Ultralytics runs a dummy forward to infer stride:
        #   m.stride = torch.tensor([s / x.shape[-2] for x in _forward(...)])
        # So _forward(...) must be iterable of Tensors. If we return (det, angle),
        # the first element is a list -> no .shape -> crash.
        #
        # Here we return ONLY det for the very first build-time forwards.
        # Once stride is set by framework, we disable this gate.
        # =========================================================
        if self._stride_infer:
            # stride 一旦被框架写入（Detect/OBB head 通常会有 stride 属性），就关闭 gate
            if getattr(self, "stride", None) is not None:
                self._stride_infer = False
            return det

        # Step4: outputs (normal training/inference)
        if self.training:
            if self.return_all_stages:
                return det, final_angle, all_angles
            return det, final_angle

        # ---------- Inference / Export ----------
        if isinstance(det, torch.Tensor):
            return torch.cat([det, final_angle], 1)

        pred = det[0]
        aux  = det[1] if len(det) > 1 else None
        out_pred = torch.cat([pred, final_angle], 1)
        return (out_pred, (aux, final_angle))

    def decode_bboxes(self, bboxes, anchors):
        return dist2rbox(bboxes, self.angle, anchors, dim=1)
