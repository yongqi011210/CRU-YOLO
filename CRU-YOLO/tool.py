# convert_ssdd_to_yolo.py
import os
import cv2
import numpy as np
from pathlib import Path

def convert_ssdd_to_yolo_obb(img_dir, ann_dir, out_dir, classes=['ship']):
    """
    将 SSDD+ 标注转换为 YOLOv8-OBB 格式
    输出每行格式: class_index x1 y1 x2 y2 x3 y3 x4 y4 (归一化 0~1)
    """
    img_dir, ann_dir, out_dir = Path(img_dir), Path(ann_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_files = list(ann_dir.glob("*.txt"))
    print(f"[INFO] 共找到 {len(ann_files)} 个标注文件，开始转换...")

    converted, skipped = 0, 0

    for ann_file in ann_files:
        with open(ann_file, 'r') as f:
            lines = f.readlines()

        out_lines = []

        # 兼容 png/jpg/jpeg
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = img_dir / (ann_file.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"[WARN] 找不到图像 {ann_file.stem}，跳过")
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 图像 {img_path} 打开失败，跳过")
            skipped += 1
            continue
        h, w = img.shape[:2]

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                print(f"[WARN] 文件 {ann_file.name} 中存在无效标注行，跳过")
                continue

            coords = list(map(float, parts[:8]))
            cls = parts[8]
            if cls not in classes:
                print(f"[WARN] 未知类别 {cls}，跳过")
                continue
            cls_id = classes.index(cls)

            # 将四个角点归一化
            norm_coords = []
            for i in range(0, 8, 2):
                x = coords[i] / w
                y = coords[i+1] / h
                norm_coords.extend([x, y])

            # 写入 YOLO OBB 格式
            out_lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in norm_coords))

        if out_lines:
            out_file = out_dir / ann_file.name
            with open(out_file, 'w') as f:
                f.write("\n".join(out_lines))
            print(f"[OK] {ann_file.name} -> {out_file.name} 转换 {len(out_lines)} 个标注")
            converted += 1
        else:
            print(f"[WARN] {ann_file.name} 没有有效标注，跳过")
            skipped += 1

    print(f"\n[INFO] 转换完成: {converted} 个文件成功, {skipped} 个文件跳过")



if __name__ == "__main__":
    # 例子
    convert_ssdd_to_yolo_obb(
        img_dir="../data/ssdd/test_inshore/img",
        ann_dir="../data/ssdd/test_inshore/label",
        out_dir="../data/ssdd+/test_inshore/labels/",
        classes=["ship"]
    )
