import torch

from ultralytics import YOLO

def main():
    model = YOLO("cascade-yolov8-obb.yaml", task='obb')
    print("Model task:", model.task)
    device = torch.device('cuda:0')
    model.model.to(device)

    model.train(
        data="ssdd.yaml",   # 数据集配置文件
        name="yolov8-obb",
        model=model,
        epochs=50,
        deterministic=False,
        imgsz=512,
        batch=4,
        device=0,
        workers=0,
        pretrained=False
    )



    model.val()

if __name__ == "__main__":
    main()
