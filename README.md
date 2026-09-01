# 21th SM High-Speed Visual Recognition Model Training

本仓库是 [21th-SM-High-Speed-Visual-Recognition](https://github.com/mx-Old-Wood/21th-SM-High-Speed-Visual-Recognition) 的模型训练附属仓库，用于整理该项目在不同阶段使用的图像分类训练与模型转换代码。实际部署、摄像头取图和赛题业务逻辑位于主仓库。

项目包含两套相互独立的历史训练流程：前期使用 TensorFlow/Keras 并导出 TFLite，后期改用 PyTorch 训练，经 ONNX 转换为 NCNN 模型。因此原来的 `NCNN` 目录已更名为 `pytorch-ncnn`，以准确表达“训练框架 + 部署格式”。

## 仓库内容

| 目录 | 阶段 | 主要用途 |
| --- | --- | --- |
| [`tensorflow/`](tensorflow/) | 前期 | TensorFlow/Keras 分类训练、检查点管理与 TFLite 导出 |
| [`pytorch-ncnn/`](pytorch-ncnn/) | 后期 | PyTorch 分类训练、评估、ONNX 导出及 NCNN 转换 |

两条流程的数据流分别为：

```text
图像目录 -> TensorFlow/Keras -> HDF5 -> TFLite
ImageFolder -> PyTorch -> ONNX -> PNNX/NCNN
```

## 发布边界

本仓库只发布训练与转换源码，不发布以下本地内容：

- 训练集、验证集、测试集及诊断样本；
- `.h5`、`.tflite`、`.pt`、`.onnx`、`.ncnn.bin`、`.ncnn.param` 等模型文件；
- 检查点、训练指标、缓存、虚拟环境和本机下载的 `pnnx` 可执行文件。

这些内容已由根目录 [`.gitignore`](.gitignore) 统一排除。克隆仓库后需要自行准备有合法使用权的数据集；训练得到的模型也只保留在本地。

## 环境与使用

两套流程应使用独立虚拟环境。项目现存环境记录表明 TensorFlow 流程曾在 Python 3.10、TensorFlow 2.10.0 下运行；PyTorch 流程的依赖版本记录在各自的 `requirements.txt` 中。

### TensorFlow/TFLite 流程

```powershell
cd tensorflow
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
```

默认训练数据目录为 `tensorflow/datasets/image/train/`。可通过 `TRAIN_DIR`、`VAL_DIR`、`IMG_H`、`IMG_W`、`NUM_CLASSES` 等环境变量覆盖配置；完整说明见 [tensorflow/README.md](tensorflow/README.md)。

### PyTorch/ONNX/NCNN 流程

```powershell
cd pytorch-ncnn
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_tiny_classifier.py --data-root dataset
```

要继续导出 NCNN，请从 [Tencent/ncnn 的 PNNX 发布页](https://github.com/pnnx/pnnx/releases) 获取工具并将 `pnnx.exe` 放入 `PATH` 或当前目录，再运行：

```powershell
.\convert_to_ncnn.ps1
```

一键训练、转换和抽样评估方式见 [pytorch-ncnn/README.md](pytorch-ncnn/README.md)。

## 与主仓库的关系

本仓库负责离线训练和模型格式转换，主仓库负责设备端推理及完整视觉系统。两者不通过 Git submodule 绑定，模型文件也不会在两个仓库间自动同步；部署前需在本地确认输入尺寸、类别顺序、预处理方式和主仓库推理代码完全一致。

## 来源与许可证

根据项目维护者提供的信息和原始下载资料，TensorFlow 流程修改自逐飞科技提供的教学脚本，PyTorch/NCNN 流程修改自龙邱科技发布的 GPL 训练工程。已核对以下来源：

- [逐飞科技相关文章（微信公众号）](https://mp.weixin.qq.com/s/kESJdQ39PskYBtFpn8QhZw)；
- [逐飞科技 LS2K0300 开源库](https://gitee.com/seekfree/LS2K0300_Library)；
- [龙邱科技相关视频（哔哩哔哩，BV1rxPNzyERk）](https://www.bilibili.com/video/BV1rxPNzyERk/)；
- [龙邱科技 Loongson 2K300/301 开源库](https://gitee.com/lq-tech/Loongson_2k300_301_Library)；
- [龙邱科技配套资源（百度网盘）](https://pan.baidu.com/s/1Yz7P73Ag9T31zfgbtxE6cA)，提取码：`tihd`。

龙邱原始训练压缩包包含 GPLv3 `LICENSE`，主要源码同时标注 `GPL-3.0-or-later`，因此 `pytorch-ncnn/` 中的派生代码继续按 GPL-3.0-or-later 发布。Tencent NCNN 本身采用 BSD 3-Clause，本仓库仅调用外部转换工具，不包含 NCNN 或 PNNX 二进制。

逐飞文章明确提供教学脚本，允许读者修改训练参数、模型层级并自行优化；但独立下载的训练资料不含许可证，文章也没有明确说明修改版源码的公开再分发或再许可条件。逐飞 Gitee 板端开源库的 GPL-3.0 不能自动视为覆盖这个单独下载的 `train.py`。因此 `tensorflow/train.py` 保留逐飞来源说明，但其上游部分的公开再分发许可仍待逐飞科技确认。

根目录 [`LICENSE`](LICENSE) 为 GPL-3.0 文本，适用于明确以 GPL-3.0-or-later 标注的下游文件及本仓库有权以该许可证发布的贡献。第三方代码仍受其原始条款约束；发布前必须补齐并核对上游信息，详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 已知限制

- 仓库不含数据和模型，克隆后无法直接复现既有结果；
- 龙邱训练流程的 GPL 许可已经确认；逐飞教学脚本的公开再分发许可仍是正式发布前的待确认项；
- 本次整理只做静态与脚本级验证，没有重新进行完整训练，也不声明任何精度或实时性能指标；
- 训练默认参数来自特定比赛场景，迁移到其他数据集时需要重新设置类别数量、输入尺寸和增强策略。
