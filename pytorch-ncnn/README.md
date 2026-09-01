# PyTorch 训练与 NCNN 导出

该目录保存项目后期的轻量分类模型流程：使用 PyTorch 从 ImageFolder 数据集训练和评估 `TinyClassifier`，导出 ONNX，再通过 `onnx2ncnn` 或 PNNX 转换为 NCNN 部署文件。

目录名使用 `pytorch-ncnn`，因为 NCNN 只是部署格式，实际训练框架是 PyTorch。原始训练包及当前派生代码采用 GPL-3.0-or-later；来源、压缩包校验值和第三方边界见仓库根目录的 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。

## 主要脚本

| 文件 | 用途 |
| --- | --- |
| `train_tiny_classifier.py` | 训练、验证、测试、ONNX 导出及指标生成 |
| `run_all.ps1` | 训练、NCNN 转换和随机抽样评估的一键流程 |
| `convert_to_ncnn.ps1` | 将已有 ONNX 模型转换为 NCNN |
| `evaluate_local_accuracy.py` | 在本地数据集上复核检查点精度 |
| `evaluate_random_subset_accuracy.py` | 随机抽样并输出分类统计 |
| `prepare_board_testset.py` | 从本地数据集中准备板端测试样本 |

默认输入目录为 `dataset/`，默认输出目录为 `artifacts/`。两者都不会提交到 Git。

## 环境

建议使用 Python 3.10 创建独立虚拟环境：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

NCNN 转换需要额外安装 `onnx2ncnn.exe` 或 `pnnx.exe`。可从 [PNNX 官方发布页](https://github.com/pnnx/pnnx/releases) 下载，并把可执行文件放入 `PATH` 或本目录。下载的工具不会提交到仓库。

## 数据格式

数据集使用 torchvision `ImageFolder` 结构，至少包含两个类别目录：

```text
dataset/
  class_a/
    image_001.jpg
  class_b/
    image_002.jpg
```

类别索引按目录名字典序生成，并写入 `artifacts/labels.txt`。部署端必须保持同一索引顺序。

## 训练

最短命令：

```powershell
python train_tiny_classifier.py --data-root dataset
```

常用参数示例：

```powershell
python train_tiny_classifier.py `
  --data-root dataset `
  --img-size 96 `
  --batch-size 64 `
  --epochs 35 `
  --lr 1e-3 `
  --weight-decay 2e-4 `
  --num-workers 4 `
  --seed 42 `
  --width-mult 0.6 `
  --patience 8 `
  --target-acc 0.95 `
  --out-dir artifacts
```

输出包括 PyTorch 检查点、ONNX、标签顺序和 JSON 指标。这些内容都属于训练产物，不纳入仓库。

## 一键训练与转换

先在 `run_all.ps1` 顶部检查数据路径、输入尺寸和训练参数，然后运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

脚本默认从当前 Conda 环境或 `PATH` 解析 Python，不再包含开发者机器上的绝对路径。`RunNcnnExport` 可控制是否执行 NCNN 转换，`RandomEval.Enabled` 可控制抽样评估。

## 单独转换已有 ONNX

```powershell
powershell -ExecutionPolicy Bypass -File .\convert_to_ncnn.ps1
```

默认读写路径为：

```text
artifacts/tiny_classifier_fp32.onnx
artifacts/tiny_classifier_fp32.ncnn.param
artifacts/tiny_classifier_fp32.ncnn.bin
```

## 独立精度复核

```powershell
python evaluate_local_accuracy.py `
  --data-root dataset `
  --checkpoint artifacts/best_model.pt `
  --img-size 96 `
  --batch-size 128 `
  --num-workers 2 `
  --seed 42
```

精度结果依赖未公开的数据集、划分种子、输入尺寸和训练参数。本仓库不发布既有模型，也不以历史本地指标作为可复现性能声明。
