# TensorFlow / TFLite 训练流程

该目录保存项目早期使用的 TensorFlow/Keras 图像分类训练脚本。脚本从按类别分目录的图片数据生成训练集和验证集，支持断点续训、数据增强、类别权重、早停及学习率调整，并可导出 HDF5 与 TFLite 模型。

此目录不包含训练集、检查点或模型文件。该脚本基于逐飞科技教学资料提供的基础 `train.py` 大幅修改；原下载包未附训练脚本许可证，其公开再分发条件尚待逐飞科技确认。详细证据和代码边界见仓库根目录的 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。

## 环境

现存本地环境记录为 Python 3.10.7、TensorFlow 2.10.0 和 NumPy 1.23.4。建议新建独立环境：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow 2.10 是最后一个支持原生 Windows GPU 的 TensorFlow 版本；其他平台或仅使用 CPU 时，应按 TensorFlow 官方兼容矩阵选择环境。

## 数据目录

默认路径为 `datasets/image/train/`，每个一级子目录代表一个类别：

```text
datasets/image/train/
  class_a/
    image_001.jpg
  class_b/
    image_002.jpg
```

默认情况下，同一目录会按 `VAL_SPLIT=0.2` 划分训练集和验证集。也可以通过 `VAL_DIR` 指定独立验证集。数据目录受 `.gitignore` 保护，不会进入 Git。

## 运行

```powershell
python train.py
```

脚本通过环境变量配置。常用变量如下：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `TRAIN_DIR` | `./datasets/image/train` | 训练数据目录 |
| `VAL_DIR` | 与 `TRAIN_DIR` 相同 | 独立验证数据目录 |
| `IMG_H` / `IMG_W` / `IMG_C` | `32` / `32` / `3` | 输入尺寸和通道数 |
| `NUM_CLASSES` | `3` | 分类数量，必须与目录一致 |
| `BATCH_SIZE` | `64` | 批大小 |
| `EPOCHS` | `400` | 最大训练轮数 |
| `LR` | `0.001` | 初始学习率 |
| `VAL_SPLIT` | `0.2` | 未指定独立验证集时的划分比例 |
| `MODEL_NAME` | `loong_cnn_model_simple.h5` | HDF5 导出路径 |
| `CKPT_DIR` | `./checkpoints/loong_cnn_model_simple` | 检查点目录 |
| `EXPORT_MODEL` | `1` | 是否导出 HDF5 模型 |
| `EXPORT_TFLITE` | `1` | 是否导出 TFLite 模型 |

PowerShell 示例：

```powershell
$env:NUM_CLASSES = "3"
$env:IMG_H = "32"
$env:IMG_W = "32"
$env:EPOCHS = "100"
python train.py
```

## 输出

训练过程中会在 `CKPT_DIR` 保存最新模型、最佳模型、指标和训练状态；训练结束后可按开关导出 `.h5` 与 `.tflite`。所有这些文件均属于本地生成物，不纳入仓库。

部署前需要同时核对类别顺序、输入尺寸、像素预处理和量化配置。仅复制模型文件而不保持这些约定，可能导致主仓库推理结果错误。
