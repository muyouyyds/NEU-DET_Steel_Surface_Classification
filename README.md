# NEU-DET Steel Surface Classification

基于 PyTorch 的钢板表面缺陷分类项目。包含训练脚本、验证脚本，以及训练过程中保存的模型权重与元数据。

## 项目文件

- `train.py`：训练模型，并在验证集上评估后保存最佳权重。
- `validation.py`：加载已训练模型，在验证集上计算准确率，并可选展示预测样例。
- `steel_surface_cnn.pt`：训练得到的模型权重文件。
- `steel_surface_cnn.json`：模型元数据，包含类别顺序和输入尺寸。

## 数据目录

项目默认使用以下目录结构：

```text
train/images/<class_name>/*.jpg
validation/images/<class_name>/*.jpg
```

类别包括：`crazing`、`inclusion`、`patches`、`pitted_surface`、`rolled-in_scale`、`scratches`。

## 快速开始

1. 训练模型

```bash
python train.py
```

2. 验证模型

```bash
python validation.py
```

3. 额外显示若干验证样本

```bash
python validation.py --show-samples 8
```

## 说明

- 训练脚本会自动保存最优模型权重。
- 验证脚本会读取 checkpoint 中的类别顺序和输入尺寸，保证推理一致性。
- 如需修改批大小、学习率或图片尺寸，可通过命令行参数调整。
