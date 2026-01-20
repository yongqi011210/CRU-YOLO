# train.py
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datasets.dataloader import build_dataloader
from models.faster_rcnn_obb import get_model  # 旋转框 Faster R-CNN
from utils.train_utils import train_one_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="SAR OBB Object Detection Training")

    # 数据集
    parser.add_argument("--dataset", type=str, default="voc", choices=["coco", "voc"],
                        help="Dataset type: coco or voc")
    parser.add_argument("--data-root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # 硬件
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 数据加载器
    train_loader, val_loader = build_dataloader(
        dataset_type=args.dataset,
        data_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True
    )

    # 模型
    model = get_model(num_classes=2)  # 背景 + ship
    model.to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 日志
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))

    # 训练循环
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, writer)
        lr_scheduler.step()
        evaluate(model, val_loader, device)

        # 保存 checkpoint
        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), ckpt_path)

    writer.close()


if __name__ == "__main__":
    main()
