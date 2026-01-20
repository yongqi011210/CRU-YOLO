import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO


# ================= 全局可视化模式开关 =================
# 可选值：
#   "tp_fp_fn" : GT主导：TP(绿 Pred) / FN(红 GT) / FP(蓝 Pred)（默认）
#   "gt"       : 只画 GT（红）
#   "pred"     : 只画预测框（蓝）
#   "gt_pred"  : GT(红) + Pred(蓝)，不区分 TP/FP/FN
VIS_MODE = "tp_fp_fn"

# ================= 阈值设置 =================
IOU_THR = 0.5           # 达到该阈值 -> 认为 GT 被检测到（画绿）
FP_DRAW_IOU_MAX = 0.10  # 预测框若与任意GT最大IoU < 该值 -> 认为是真误检FP（画蓝）
# 解释：如果某个pred和GT有明显重叠但没达阈值（定位差），它会造成红+蓝混乱。
# 你要“没达到阈值就是没达到阈值”，所以这类pred我们不画（不算FP可视化）。
# 建议：0.05~0.15，默认0.10较干净

DRAW_IOU_TEXT = False   # 默认不显示 IoU 文本


# =========================================================
# 多边形 IoU（更鲁棒：buffer(0)修复无效 polygon）
# =========================================================
def poly_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    try:
        p1 = Polygon(poly1).buffer(0)
        p2 = Polygon(poly2).buffer(0)
    except Exception:
        return 0.0

    if (not p1.is_valid) or (not p2.is_valid):
        return 0.0

    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return float(inter / union) if union > 0 else 0.0


def to_px(poly_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = poly_norm.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts.astype(int)


# =========================================================
# GT主导匹配：
#   对每个GT找一个“最佳pred”，若IoU>=thr -> TP(画绿pred) 否则 FN(画红GT)
#   剩余pred：若与任意GT maxIoU < FP_DRAW_IOU_MAX -> FP(画蓝)
# =========================================================
def match_by_gt(
    gt_polys: np.ndarray,
    pred_polys: np.ndarray,
    iou_thr: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    返回：
      tp_matches: [(gt_i, pred_j, iou), ...]
      fn_gts    : [gt_i, ...]
      leftover_preds: [pred_j, ...]  # 未被用于TP的预测
    """
    nG = len(gt_polys)
    nP = len(pred_polys)

    used_pred = set()
    tp_matches = []
    fn_gts = []

    for gi in range(nG):
        best_j = -1
        best_iou = 0.0

        for pj in range(nP):
            if pj in used_pred:
                continue
            iou = poly_iou(gt_polys[gi], pred_polys[pj])
            if iou > best_iou:
                best_iou = iou
                best_j = pj

        if best_j >= 0 and best_iou >= iou_thr:
            tp_matches.append((gi, best_j, float(best_iou)))
            used_pred.add(best_j)
        else:
            fn_gts.append(gi)

    leftover_preds = [pj for pj in range(nP) if pj not in used_pred]
    return tp_matches, fn_gts, leftover_preds


def filter_true_fp(
    gt_polys: np.ndarray,
    pred_polys: np.ndarray,
    leftover_preds: List[int],
    fp_draw_iou_max: float,
) -> List[int]:
    """
    只把“远离所有GT”的pred当作可视化FP。
    对那些与某个GT有明显重叠但没达阈值的pred（定位差），不画，避免红+蓝混乱。
    """
    true_fp = []
    for pj in leftover_preds:
        max_iou = 0.0
        for gi in range(len(gt_polys)):
            iou = poly_iou(gt_polys[gi], pred_polys[pj])
            if iou > max_iou:
                max_iou = iou

        if max_iou < fp_draw_iou_max:
            true_fp.append(pj)
    return true_fp


# =========================================================
# 画图
# =========================================================
def draw_vis(image_path, pred_path, gt_path, save_path, mode="tp_fp_fn", iou_thr=0.5):
    if not os.path.exists(image_path):
        print(f"[WARN] image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] failed to read image: {image_path}")
        return

    h, w = img.shape[:2]

    # 读取 GT
    gt_polys_list = []
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                coords = list(map(float, parts[1:9]))
                poly = np.array(coords, dtype=float).reshape(4, 2)
                gt_polys_list.append(poly)
    else:
        print(f"[INFO] no GT label for: {image_path}")

    # 读取 Pred
    pred_polys_list = []
    if os.path.exists(pred_path):
        with open(pred_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                coords = list(map(float, parts[1:9]))
                poly = np.array(coords, dtype=float).reshape(4, 2)
                pred_polys_list.append(poly)
    else:
        print(f"[INFO] no PRED label for: {image_path}")

    gt_polys = np.array(gt_polys_list, dtype=float) if len(gt_polys_list) > 0 else np.zeros((0, 4, 2), dtype=float)
    pred_polys = np.array(pred_polys_list, dtype=float) if len(pred_polys_list) > 0 else np.zeros((0, 4, 2), dtype=float)

    # 只画 GT
    if mode == "gt":
        for g in gt_polys:
            cv2.polylines(img, [to_px(g, w, h)], True, (0, 0, 255), 2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        print(f"[SAVE][GT] {save_path}")
        return

    # 只画 Pred
    if mode == "pred":
        for p in pred_polys:
            cv2.polylines(img, [to_px(p, w, h)], True, (255, 0, 0), 2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        print(f"[SAVE][PRED] {save_path}")
        return

    # GT + Pred
    if mode == "gt_pred":
        for g in gt_polys:
            cv2.polylines(img, [to_px(g, w, h)], True, (0, 0, 255), 2)
        for p in pred_polys:
            cv2.polylines(img, [to_px(p, w, h)], True, (255, 0, 0), 2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        print(f"[SAVE][GT+PRED] {save_path}")
        return

    # ========== GT主导 TP/FN/FP ==========
    tp_matches, fn_gts, leftover_preds = match_by_gt(gt_polys, pred_polys, iou_thr=iou_thr)
    fp_preds = filter_true_fp(gt_polys, pred_polys, leftover_preds, fp_draw_iou_max=FP_DRAW_IOU_MAX)

    # 绘制顺序：FN(红GT) -> FP(蓝Pred) -> TP(绿Pred)
    # 1) FN：只画GT红
    for gi in fn_gts:
        cv2.polylines(img, [to_px(gt_polys[gi], w, h)], True, (0, 0, 255), 2)

    # 2) FP：只画Pred蓝（仅远离GT的真误检）
    for pj in fp_preds:
        cv2.polylines(img, [to_px(pred_polys[pj], w, h)], True, (255, 0, 0), 2)

    # 3) TP：只画Pred绿
    for gi, pj, miou in tp_matches:
        pts = to_px(pred_polys[pj], w, h)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        if DRAW_IOU_TEXT:
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(img, f"IoU {miou:.2f}", (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    print(f"[SAVE][GT-driven TP/FN/FP] {save_path}")


# =========================================================
# 主流程：先跑 YOLO predict，再做可视化
# =========================================================
def main():
    model = YOLO(
        "E:/mul/YOLO_datasets/runs/obb/yolo12-obb/weights/best.pt",
        task="obb",
    )

    results = model.predict(
        source="E:/mul/data/RSDD/test_inshore/images/",
        save=True,
        save_txt=True,
        save_conf=False,
        show_labels=False,
        show_conf=True,
        imgsz=512,
        device="cuda:0",
    )

    save_dir = Path(results[0].save_dir)
    pred_dir = save_dir / "labels"
    img_out_dir = save_dir

    gt_root = Path("E:/mul/data/RSDD/test_inshore/labels")

    print(f"[INFO] Visualization mode = {VIS_MODE}")
    print(f"[INFO] IOU_THR={IOU_THR}, FP_DRAW_IOU_MAX={FP_DRAW_IOU_MAX}, DRAW_IOU_TEXT={DRAW_IOU_TEXT}")

    for r in results:
        src_path = Path(r.path)
        name = src_path.stem

        img_path = None
        cand1 = img_out_dir / src_path.name
        if cand1.exists():
            img_path = cand1
        else:
            for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                cand = img_out_dir / f"{name}{ext}"
                if cand.exists():
                    img_path = cand
                    break

        if img_path is None:
            print(f"[WARN] no predicted image found for {src_path.name}")
            continue

        pred_txt = pred_dir / f"{name}.txt"
        gt_txt = gt_root / f"{name}.txt"
        save_path = img_out_dir / f"{name}_vis.jpg"

        draw_vis(
            image_path=str(img_path),
            pred_path=str(pred_txt),
            gt_path=str(gt_txt),
            save_path=str(save_path),
            mode=VIS_MODE,
            iou_thr=IOU_THR,
        )

    print("[DONE] 全部预测与可视化完成！")


if __name__ == "__main__":
    main()
