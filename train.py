from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 这是训练脚本的入口文件。
# 功能：
# - 使用 ImageFolder 从按类组织的文件夹中读取训练/验证图像
# - 定义一个简单的卷积神经网络 `SteelCNN`
# - 训练模型并在验证集上评估、保存最佳 checkpoint（.pt）和元数据（.json）
#
# 说明：
# - 期望目录结构：
#   train/images/<class_name>/*.jpg
#   validation/images/<class_name>/*.jpg
# - 保存的 checkpoint 包含 model_state_dict、classes 和 image_size
# - 验证脚本 validation.py 会依赖 checkpoint 中的 classes 与 image_size


CLASSES = [
	"crazing",
	"inclusion",
	"patches",
	"pitted_surface",
	"rolled-in_scale",
	"scratches",
]


class SteelCNN(nn.Module):
	"""
	一个小型的卷积神经网络，用于图像分类。

	结构说明：
	- 多层 Conv2d + BatchNorm + ReLU + MaxPool 用作特征提取
	- 最后使用 AdaptiveAvgPool 将特征图尺寸归一化到 (1,1)
	- 使用一个全连接层输出类别分数
	设计目标是满足中小规模图像分类任务，参数量较小，方便在 GPU 上训练。
	"""

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
		# 前向传播：输入 (B,3,H,W)，输出 (B,num_classes)
		x = self.features(x)
		return self.classifier(x)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
	"""
	在给定 dataloader 上评估模型准确率。

	说明：
	- 该函数在 `torch.no_grad()` 下运行以节省显存与计算
	- 返回值是 0-1 之间的小数，表示正确分类比例
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


def train(args: argparse.Namespace) -> None:
	# 设置随机种子，保证可复现性（在单机单卡/CPU上有限）
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)

	# 选择设备（优先 GPU）
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_root = Path(args.train_dir)
	validation_root = Path(args.validation_dir)

	# 数据增广与归一化（训练/验证使用不同的 transform）
	train_transform = transforms.Compose(
		[
			transforms.Resize((args.image_size, args.image_size)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	validation_transform = transforms.Compose(
		[
			transforms.Resize((args.image_size, args.image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)

	# 使用 ImageFolder 从磁盘读取图像并自动根据子文件夹名生成类标签
	train_dataset = datasets.ImageFolder(root=str(train_root), transform=train_transform)
	validation_dataset = datasets.ImageFolder(root=str(validation_root), transform=validation_transform)

	# DataLoader 用于批量读取数据
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=0,
		pin_memory=torch.cuda.is_available(),
	)
	validation_loader = DataLoader(
		validation_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=0,
		pin_memory=torch.cuda.is_available(),
	)

	# 检查数据集中的类别顺序是否与 CLASSES 常量一致（仅用于提示）
	if train_dataset.classes != CLASSES:
		print("检测到的类别顺序:", train_dataset.classes)

	# 构建模型、损失函数、优化器与学习率调度
	model = SteelCNN(num_classes=len(train_dataset.classes)).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.5)

	best_validation_accuracy = 0.0
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# 训练循环
	for epoch in range(args.epochs):
		model.train()
		running_loss = 0.0
		correct = 0
		total = 0

		for images, labels in train_loader:
			images = images.to(device)
			labels = labels.to(device)

			optimizer.zero_grad(set_to_none=True)
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# 统计训练损失与准确率
			running_loss += loss.item() * labels.size(0)
			predictions = outputs.argmax(dim=1)
			correct += (predictions == labels).sum().item()
			total += labels.size(0)

		# 学习率调度在每个 epoch 结束后更新
		scheduler.step()

		train_loss = running_loss / total if total else 0.0
		train_accuracy = correct / total if total else 0.0
		validation_accuracy = evaluate(model, validation_loader, device)

		print(
			f"Epoch {epoch + 1}/{args.epochs} | "
			f"loss={train_loss:.4f} | "
			f"train_acc={train_accuracy:.4f} | "
			f"val_acc={validation_accuracy:.4f}"
		)

		# 如果验证集准确率有提升，则保存 checkpoint（覆盖旧的）
		if validation_accuracy >= best_validation_accuracy:
			best_validation_accuracy = validation_accuracy
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"classes": train_dataset.classes,
					"image_size": args.image_size,
				},
				output_path,
			)

	# 保存训练元信息到 JSON 便于后续加载
	metadata_path = output_path.with_suffix(".json")
	metadata_path.write_text(
		json.dumps(
			{
				"classes": train_dataset.classes,
				"image_size": args.image_size,
				"best_validation_accuracy": best_validation_accuracy,
			},
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)

	print(f"模型已保存到: {output_path}")
	print(f"最佳验证准确率: {best_validation_accuracy:.4f}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a CNN for steel surface label recognition.")
	parser.add_argument("--train-dir", default="train/images", help="训练集图片目录")
	parser.add_argument("--validation-dir", default="validation/images", help="验证集图片目录")
	parser.add_argument("--output", default="steel_surface_cnn.pt", help="模型保存路径")
	parser.add_argument("--epochs", type=int, default=60, help="训练轮数")
	parser.add_argument("--batch-size", type=int, default=64, help="批大小")
	parser.add_argument("--learning-rate", type=float, default=0.0005, help="学习率")
	parser.add_argument("--image-size", type=int, default=224, help="输入图片尺寸")
	parser.add_argument("--seed", type=int, default=42, help="随机种子")
	return parser.parse_args()


if __name__ == "__main__":
	train(parse_args())
