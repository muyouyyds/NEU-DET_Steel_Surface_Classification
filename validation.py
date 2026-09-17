from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import random

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 验证与可视化脚本。
# 功能：
# - 加载 train.py 训练生成的 checkpoint（.pt），并用 `SteelCNN` 构建模型加载权重
# - 计算验证集准确率
# - 可选：弹出窗口展示若干验证图片并显示预测类别与真实类别（由 --show-samples 控制）
#
# 说明：如果在无头服务器（没有显示器）上运行，`plt.show()` 可能失败，
# 可将图像保存到文件或在本地环境运行以弹窗查看。


def imshow_tensor(img_tensor: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    将张量格式的图像转换为可由 matplotlib 显示的 numpy 数组。

    img_tensor: Tensor 格式为 (C,H,W)，范围为标准化后的值（经过 ToTensor 和 Normalize）
    mean/std: 与训练时相同的归一化参数，用来反归一化回 [0,1]
    返回值：H x W x C 的 numpy 数组
    """
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    return img


def show_samples(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    device: torch.device,
    classes,
    num_samples: int = 8,
    image_size: int = 224,
    output_dir: Path | None = None,
):
    """
    在验证集中随机抽取若干样例图片，反归一化后展示并标出真实/预测标签。

    注意：dataset[idx] 返回的是已经经过 transform 的 Tensor，因此此处直接传入 model 进行预测，
    并用 imshow_tensor 反归一化后显示。
    """
    model.eval()
    if len(dataset) <= num_samples:
        indices = list(range(len(dataset)))
    else:
        indices = random.sample(range(len(dataset)), k=num_samples)

    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(cols * 3.8, rows * 3.8))
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, label = dataset[idx]
            inp = img.unsqueeze(0).to(device)
            out = model(inp)
            pred = int(out.argmax(dim=1).cpu().item())

            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(imshow_tensor(img))
            ax.set_title(f"Sample {i + 1}", fontsize=10, pad=8)
            ax.text(
                0.5,
                -0.12,
                f"True: {classes[label]} | Pred: {classes[pred]}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2.5),
            )
            ax.axis("off")

    figure.tight_layout()

    if output_dir is not None:
        output_path = output_dir / f"validation_samples_{num_samples}.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"验证样本已保存到: {output_path}")
    else:
        plt.show()

    plt.close(figure)


class SteelCNN(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    在 dataloader 上计算模型准确率，返回 0-1 之间的小数。
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a trained steel surface CNN.")
    parser.add_argument("--validation-dir", default="validation/images", help="验证集图片目录")
    parser.add_argument("--model-path", default="steel_surface_cnn.pt", help="模型文件路径")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--image-size", type=int, default=224, help="输入图片尺寸")
    parser.add_argument("--show-samples", type=int, default=0, help="展示若干验证图片及预测标签，0 表示不展示")
    args = parser.parse_args()

    # 加载模型 checkpoint（包含权重与元信息）并构造模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)

    image_size = checkpoint.get("image_size", args.image_size)
    validation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    validation_dataset = datasets.ImageFolder(root=str(Path(args.validation_dir)), transform=validation_transform)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # 根据 checkpoint 内记录的 classes 数目构建模型并加载权重
    model = SteelCNN(num_classes=len(checkpoint.get("classes", validation_dataset.classes)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    accuracy = evaluate(model, validation_loader, device)
    print(f"验证集正确率: {accuracy * 100:.2f}%")

    if args.show_samples and args.show_samples > 0:
        try:
            show_samples(
                model,
                validation_dataset,
                device,
                checkpoint.get("classes", validation_dataset.classes),
                args.show_samples,
                image_size,
                Path("validation_samples"),
            )
        except Exception as e:
            print(f"展示样例时出错: {e}")


if __name__ == "__main__":
    main()
