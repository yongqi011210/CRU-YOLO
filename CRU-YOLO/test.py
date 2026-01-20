# train_yolov8_ssdd.py
from ultralytics import YOLO

def main():
    # 加载预训练 OBB 模型 (也可以换 yolov8s-obb.pt, yolov8m-obb.pt)
    model = YOLO("yolov8n-obb.yaml")
    print("Model task:", model.task)  # 应该输出 obb

    # 训练
    model.train(
        data="ssdd.yaml",
        model=model,
        epochs=50,
        imgsz=512,
        batch=4,  # 建议先小一点，避免 OOM
        device=0,
        workers=0,
        pretrained=False
    )

    # 评估
    model.val()



if __name__ == "__main__":
    main()
