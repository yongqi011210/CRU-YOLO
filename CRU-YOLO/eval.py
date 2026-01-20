import torch
from ultralytics import YOLO

def main():
    # ===============================
    # 1️⃣ 加载模型
    # ===============================
    model_path = "E:/mul/YOLO_datasets/runs/obb/yolo12-obb/weights/best.pt"   # 修改为你实际训练的权重路径
    data_path = "eval.yaml"                            # 数据配置文件（含 test/val 路径）
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"[INFO] Using device: {device}")
    model = YOLO(model_path, task='obb').to(device)
    print(f"[INFO] Loaded model: {model_path}")

    # ===============================
    # ❗禁用类别名称和置信度
    # ===============================
    model.overrides['show_labels'] = False
    model.overrides['show_conf'] = False

    # ===============================
    # 2️⃣ 执行评估（验证/测试）
    # ===============================
    results = model.val(
        data=data_path,        # 数据集配置文件
        imgsz=512,             # 测试图像尺寸
        batch=4,               # 批量大小
        save=False,             # 保存检测结果图像
        save_txt=False,         # 保存预测框（YOLO txt 格式）
        save_hybrid=False,     # 是否保存混合标签
        verbose=True,          # 打印详细结果
        plots=True,            # 保存 PR 曲线、混淆矩阵等
        visualize=False,         # 可视化检测结果
        device=device,
        # =============== 仿 mmrotate test_cfg ==================
        # conf=0.1,  # rcnn.score_thr
        # iou=0.5,  # rcnn.nms.iou_thr
        # max_det=2000,  # rcnn.max_per_img
    )
    rd = results.results_dict
    p = rd["metrics/precision(B)"]
    r = rd["metrics/recall(B)"]
    map50 = rd["metrics/mAP50(B)"]
    map5095 = rd["metrics/mAP50-95(B)"]

    # 2) 算 F1
    f1 = 2 * p * r / (p + r + 1e-16)

    # # 3) 从 metrics（也就是 OBBMetrics 实例）里取我们在 get_stats 里挂上的 fpr / ffpi
    # fpr = getattr(results, "fpr", None)
    # ffpi = getattr(results, "ffpi", None)
    #
    # print("[INFO] Evaluation completed.")
    # print(f"mAP50: {map50:.4f}")
    # print(f"mAP50-95: {map5095:.4f}")
    # print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1-score: {f1:.4f}")

    # if fpr is not None:
    #     # fpr 是 0~1，小数，这里转成百分比输出
    #     print(f"FPR: {fpr * 100:.2f}%")
    # else:
    #     print("FPR: <not computed>")
    #
    # if ffpi is not None:
    #     print(f"FFPI: {ffpi:.4f}")
    # else:
    #     print("FFPI: <not computed>")

if __name__ == "__main__":
    main()
