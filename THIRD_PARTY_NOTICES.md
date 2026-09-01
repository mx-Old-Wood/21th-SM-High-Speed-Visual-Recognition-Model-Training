# 第三方来源与许可证说明

本仓库包含基于第三方训练项目修改而来的代码。本文件依据现有本地资料和项目维护者提供的信息记录可核实的来源、修改关系与许可证边界，但不能替代各上游项目的原始许可证文本。

## TensorFlow 训练流程

- 位置：`tensorflow/train.py`
- 上游来源：逐飞科技教程配套的基础 TensorFlow 训练脚本
- 教程：[逐飞基于 LS2K0300 核心板的 AI 视觉教程](https://mp.weixin.qq.com/s/kESJdQ39PskYBtFpn8QhZw)
- 相关设备端仓库：[seekfree/LS2K0300_Library](https://gitee.com/seekfree/LS2K0300_Library)，Gitee 标识其许可证为 GPL-3.0
- 原始下载文件：`龙芯AI视觉相关资料.7z`，SHA-256：`6C493062E432CC620E1F1B22734F34D40CCA83D147866FBC60A8A88F99FCA42F`
- 原始脚本：`模型训练资料/train.py`，SHA-256：`8190DAB996E9120BEDF207496D3FDE38C43DEFEF0618ABCBAA64C22F322F95A7`
- 本仓库用途：项目前期的图像分类训练、检查点管理和 TFLite 导出
- 使用与修改依据：教程明确提供该脚本用于演示，并引导读者修改训练参数、模型层级和自行优化
- 许可状态：独立下载脚本的公开再分发与再许可条件未被明确说明

逐飞教程说明其提供了一个用于演示具体流程的基础训练脚本，要求读者自行修改训练参数和模型层级，并鼓励继续进行数据增强、模型量化和二次训练等优化。下载的压缩包包含该脚本和示例数据，但未附带 `LICENSE`、README、版权声明或再分发条款。另行链接的 `LS2K0300_Library` 采用 GPL-3.0，但教程将“模型训练相关文件”和“开源库”列为两个独立下载项，因此本仓库不假定设备端仓库的 GPL-3.0 自动覆盖单独下载的 `train.py`。

本仓库中的脚本已被大幅修改，并保留了来源说明。

TensorFlow 仅作为依赖安装，本仓库未包含其源码或二进制文件。其许可证以 [TensorFlow 官方项目](https://github.com/tensorflow/tensorflow)公布的内容为准。

## PyTorch 至 NCNN 训练流程

- 位置：`pytorch-ncnn/`
- 上游来源：龙邱科技发布的 TinyClassifier 训练项目
- 教程：[龙芯智能车轻量级分类模型训练教程（BV1rxPNzyERk）](https://www.bilibili.com/video/BV1rxPNzyERk/)
- 相关官方仓库：[lq-tech/Loongson_2k300_301_Library](https://gitee.com/lq-tech/Loongson_2k300_301_Library)，Gitee 标识其许可证为 GPL-3.0，并链接了同一份训练资料
- 原始下载：[百度网盘](https://pan.baidu.com/s/1Yz7P73Ag9T31zfgbtxE6cA)，提取码：`tihd`
- 原始压缩包：`LQ_TinyClassifier-master.zip`，SHA-256：`212D79C54DE7613B12B0225885BEDBB5D2348336FFB06C1D347790EBE35BBF9C`
- 本仓库用途：项目后期采用的 PyTorch 训练与评估、ONNX 导出和 NCNN 转换流程
- 许可证：GPL-3.0-or-later；原始压缩包根目录包含完整 GPLv3 `LICENSE`，主要源码同时带有 SPDX 许可证标识
- 原始版权声明：`Copyright (C) 2026 望着天空的眼睛 <a15234181830@163.com>`

本仓库保留了原有的 GPL 许可证声明和版权声明，并标明相关文件已经修改。在遵守 GPL-3.0-or-later 的前提下，这些派生文件可以源码形式再分发；发布时应保留原始声明、标明修改并提供相应源码。

PyTorch、torchvision、ONNX、ONNX Runtime 及相关 Python 软件包仅作为依赖安装，本仓库未包含其源码或二进制文件，各依赖继续适用其各自的许可证。

## NCNN 与 PNNX

转换脚本可以调用 `onnx2ncnn` 或 `pnnx`。本仓库明确排除另行下载的转换器二进制文件。[Tencent/ncnn](https://github.com/Tencent/ncnn) 的源码和二进制发行版采用 BSD 3-Clause 许可证，其中单独列出的第三方组件继续适用各自的许可证。PNNX 是 [Tencent/ncnn](https://github.com/Tencent/ncnn/tree/master/tools/pnnx) 项目的一部分；应从官方项目获取二进制文件及其完整适用声明。

## 相关应用仓库

训练生成的模型用于配套项目 [mx-Old-Wood/21th-SM-High-Speed-Visual-Recognition](https://github.com/mx-Old-Wood/21th-SM-High-Speed-Visual-Recognition)。该项目是独立作品，具有自己的发布范围与许可证边界。
