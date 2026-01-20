import torch
import torch.nn as nn
import math
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import dist2rbox


class CascadeROIOBB(Detect):
    """
    Cascade ROI-based OBB detection head (修正版：提前计算角度以避免 self.angle 丢失)
    - 在调用 super().forward(x) 之前完成 cascade angle 预测并在 eval 时设置 self.angle
    - 保留训练/推理输出兼容逻辑与丰富 debug 信息
    """

    def __init__(self, nc=80, ne=1, ch=(), cascade_stages=2):
        super().__init__(nc, ch)
        self.ne = ne
        self.cascade_stages = cascade_stages

        # print(f"[DEBUG] 初始化 CascadeROIOBB, 输入通道 ch={ch}, 级联阶段={cascade_stages}")

        c4 = max(ch[0] // 4, self.ne)
        self.roi_branches = nn.ModuleList()
        for stage_idx in range(self.cascade_stages):
            stage_branch = nn.ModuleList()
            for i, xch in enumerate(ch):
                # print(f"[DEBUG] 构建 Stage {stage_idx} - 分支 {i}: 输入通道={xch}, 中间通道={c4}")
                stage_branch.append(nn.Sequential(
                    Conv(xch, c4, 3),
                    Conv(c4, c4, 3),
                    nn.Conv2d(c4, self.ne, 1)
                ))
            self.roi_branches.append(stage_branch)

    def forward(self, x):
        bs = x[0].shape[0]
        # print(f"[DEBUG] 前向传播 CascadeROIOBB, 输入层数={len(x)}")
        # for i, xi in enumerate(x):
        #     # print(f"[DEBUG] 输入特征图 {i}: shape={tuple(xi.shape)}")

        # 保留父类可能会改变的原始特征（避免后续通道不匹配）
        x_raw = [xi.clone() for xi in x]

        # ---------- 先计算 cascade angles（在调用 Detect.forward 之前） ----------
        all_angles = []
        for stage_idx in range(self.cascade_stages):
            # print(f"[DEBUG] 预计算 ========== Cascade Stage {stage_idx} ==========")
            angle_list = []
            for i in range(self.nl):
                xi = x_raw[i]
                # print(f"[DEBUG] 预计算 Stage {stage_idx}, 分支 {i} 输入 shape={tuple(xi.shape)}")
                try:
                    out = self.roi_branches[stage_idx][i](xi)
                except Exception as e:
                    # print(f"[ERROR] 预计算 Stage {stage_idx} 分支 {i} forward 出错: {e}")
                    raise
                # print(f"[DEBUG] 预计算 Stage {stage_idx}, 分支 {i} 输出 shape={tuple(out.shape)}")
                angle_list.append(out.view(bs, self.ne, -1))
            angle_stage = torch.cat(angle_list, 2)
            angle_stage = (angle_stage.sigmoid() - 0.25) * math.pi
            all_angles.append(angle_stage)
            # print(f"[DEBUG] 预计算 Stage {stage_idx} 合并角度输出 shape={tuple(angle_stage.shape)}")

        final_angle = all_angles[-1]

        # 若是 eval 模式，提前把 angle 赋给 self，这样 Detect.forward / decode_bboxes 可安全使用 self.angle
        if not self.training:
            self.angle = final_angle
            # print(f"[DEBUG] 非训练模式：已设置 self.angle，shape={tuple(final_angle.shape)}")

        # ---------- 现在安全调用父类 forward（父类可能会在内部使用 decode_bboxes()） ----------
        # print(f"[DEBUG] 调用 Detect.forward()（现在可安全使用 self.angle）开始")
        base_out = super().forward(x)
        # print(f"[DEBUG] Detect.forward() 完成, base_out type: {type(base_out)}")

        # ---------- 构造训练/推理返回格式 ----------
        if self.training:
            # print("[DEBUG] 训练模式：根据 base_out 类型构造 preds")

            if isinstance(base_out, (list, tuple)):
                # base_out 是 list/tuple，loss 期望 preds = (list_of_layer_preds, pred_angle_tensor)
                # 因此返回 (base_out, final_angle)
                # print(f"[DEBUG] base_out 是 list/tuple (len={len(base_out)}). 返回 (base_out, final_angle).")
                return (base_out, final_angle)
            else:
                # base_out 已是 tensor（例如 [bs, N, C]），loss 期望 preds[1] = (feats, pred_angle)
                feats_tensor = base_out
                # print(f"[DEBUG] base_out 为 tensor, shape={tuple(feats_tensor.shape)}. 返回 (base_out, (feats, final_angle)).")
                return (base_out, (feats_tensor, final_angle))

        else:
            # print("[DEBUG] 推理模式输出最终角度/拼接结果")
            if self.export:
                return torch.cat([base_out, final_angle], 1)
            else:
                # 保持与 Detect 的输出兼容： (outs_with_bbox, (aux, angle))
                try:
                    return torch.cat([base_out[0], final_angle], 1), (base_out[1], final_angle)
                except Exception:
                    # 兜底：如果 base_out 不是 list-like，则简单返回
                    return base_out, (base_out, final_angle)

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        # 使用 self.angle（在 eval 时已被提前设置）
        return dist2rbox(bboxes, self.angle, anchors, dim=1)