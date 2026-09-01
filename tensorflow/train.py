# Based on the basic TensorFlow training script supplied with SeekFree
# Technology's LS2K0300 AI vision tutorial. Substantially modified in 2026.
# Upstream redistribution terms were not included with the original archive;
# see ../THIRD_PARTY_NOTICES.md for provenance and licensing boundaries.

import tensorflow as tf
from tensorflow.keras import layers, models, Sequential, utils,optimizers, losses
import os
import json
import numpy as np
import random

print("TensorFlow版本：", tf.__version__)
silence_tf_warnings = os.getenv("SILENCE_TF_WARNINGS", "1") != "0"
if silence_tf_warnings:
    tf.get_logger().setLevel("ERROR")
gpus = tf.config.list_physical_devices("GPU")
print("可用GPU：", gpus)
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


# ================== 可调参数（支持环境变量覆盖） ================== #
# TRAIN_DIR: 训练集目录（目录结构需为: <dir>/<class_name>/*）
train_data_dir = os.getenv("TRAIN_DIR", "./datasets/image/train")
# VAL_DIR: 验证集目录；不设置时默认与训练集一致（此时使用 VAL_SPLIT 从 TRAIN_DIR 里划分验证集）
validation_data_dir = os.getenv("VAL_DIR", train_data_dir)

# ================== 图象参数 ================== #
#需与数据一致,否则可能导致训练失败或性能不佳
img_height = int(os.getenv("IMG_H", "32"))                                                      #图象高度
img_width = int(os.getenv("IMG_W", "32"))                                                       #图象宽度
img_depth = int(os.getenv("IMG_C", "3"))                                                        #图象通道数（RGB为3，灰度图为1）

num_classes = int(os.getenv("NUM_CLASSES", "3"))                                                #类别数量（必须与你的目录类别数一致）

# ================== 训练参数 ================== #
batch_size = int(os.getenv("BATCH_SIZE", "64"))                                                #批大小（根据数据集大小和GPU内存调整）
epochs = int(os.getenv("EPOCHS", "400"))                                                        #训练轮数（根据数据集大小和过拟合情况调整）
learning_rate = float(os.getenv("LR", "0.001"))                                                #学习率（根据模型复杂度和数据集调整）
validation_split = float(os.getenv("VAL_SPLIT", "0.2"))                                         #验证集比例（当 VAL_DIR == TRAIN_DIR 时使用）

validation_freq = int(os.getenv("VAL_FREQ", "5"))                                               # VAL_FREQ: 多少个 epoch 做一次验证（影响 val_* 指标、早停与 best_model 判优）

seed = int(os.getenv("SEED", "123"))                                                            # SEED: 数据集划分的随机种子

# ========== 模型与检查点设置 ========== #
# MODEL_NAME/CKPT_DIR: 导出与断点续训
Model_name = os.getenv("MODEL_NAME", "loong_cnn_model_simple.h5")                               #最终模型文件名（HDF5格式，后缀建议用 .h5）
checkpoint_dir = os.getenv("CKPT_DIR", "./checkpoints/loong_cnn_model_simple")                  #断点续训与模型权重保存目录（将自动在此目录下保存 latest_model.h5 和 best_model.h5）

# BEST_MONITOR/BEST_FALLBACK_MONITOR: 保存最优模型（优先使用 val_*，没有验证时用训练指标兜底）
best_monitor = os.getenv("BEST_MONITOR", "val_accuracy")                                        #监控指标（如 val_accuracy、val_loss、accuracy 等；建议优先使用验证指标）
best_fallback_monitor = os.getenv("BEST_FALLBACK_MONITOR", "accuracy")                          #兜底监控指标（当 BEST_MONITOR 不存在时使用，如 accuracy；建议使用训练指标）
best_allow_fallback_env = os.getenv("BEST_ALLOW_FALLBACK")                                      #兜底监控指标开关（当 BEST_MONITOR 不存在时是否允许使用 BEST_FALLBACK_MONITOR；默认根据 BEST_MONITOR 是否以 "val_" 开头自动判断：如果 BEST_MONITOR 是验证指标则默认允许兜底，否则默认不允许兜底）
if best_allow_fallback_env is None:
    best_allow_fallback = 0 if best_monitor.startswith("val_") else 1
else:
    best_allow_fallback = best_allow_fallback_env != "0"
best_verbose = os.getenv("BEST_VERBOSE", "1") != "0"

# ========== 早停 ========== #
# EARLYSTOP_*: 早停（EarlyStopping）
early_stop_enabled = os.getenv("EARLYSTOP", "1") != "0"                                         #早停开关（0=关闭；1=启用）
early_stop_monitor = os.getenv("EARLYSTOP_MONITOR", best_monitor)                               #早停监控指标
early_stop_patience = int(os.getenv("EARLYSTOP_PATIENCE", "10"))                                #早停耐心值
early_stop_min_delta = float(os.getenv("EARLYSTOP_MIN_DELTA", "0.0"))                           #早停最小增量
early_stop_mode = os.getenv("EARLYSTOP_MODE", "auto")                                           #早停模式
early_stop_restore_best_weights = os.getenv("EARLYSTOP_RESTORE_BEST", "1") != "0"               #早停是否恢复最佳权重

# ========== 学习率调度（LR_SCHED） ========== #
# 学习率调度（LR_SCHED）：none/plateau/cosine
lr_sched = os.getenv("LR_SCHED", "plateau").strip().lower()

# plateau: ReduceLROnPlateau（监控指标长期不提升则降低学习率）
lr_plateau_monitor = os.getenv("LR_PLATEAU_MONITOR", "val_loss")                                #监控指标（如 val_loss、val_accuracy、accuracy 等；建议优先使用验证指标）
lr_plateau_factor = float(os.getenv("LR_PLATEAU_FACTOR", "0.5"))                                #学习率降低因子（每次降低为原来的多少，如 0.5 表示降低到一半）
lr_plateau_patience = int(os.getenv("LR_PLATEAU_PATIENCE", "2"))                                #耐心值（多少个 epoch 监控指标不提升后降低学习率）
lr_plateau_min_lr = float(os.getenv("LR_PLATEAU_MIN_LR", "0.000001"))                           #学习率下限（降低学习率时的最小值，避免过低导致训练停滞）
lr_plateau_cooldown = int(os.getenv("LR_PLATEAU_COOLDOWN", "0"))                                #冷却时间（每次降低学习率后多少个 epoch 内不再监控指标提升，以避免过快连续降低学习率）

# cosine: 余弦退火（按 epoch 变化；LR 为 LR_COSINE_MIN_LR 到 LR 之间）
lr_cosine_min_lr = float(os.getenv("LR_COSINE_MIN_LR", "0.000001"))                             #余弦退火最小学习率

# ================== 数据增强（AUGMENT） ================== #
augment_enabled = os.getenv("AUGMENT", "1") != "0"                                              #数据增强开关（0=关闭；1=启用）
aug_flip_enabled = os.getenv("AUG_FLIP", "0") != "0"                                            #翻转开关（0=不翻转；1=翻转；当启用时使用 AUG_FLIP_MODE 指定的翻转模式进行随机翻转）
aug_flip_mode = os.getenv("AUG_FLIP_MODE", "horizontal").strip().lower()                        #翻转模式（none/0=不翻转；horizontal=水平翻转；vertical=垂直翻转；both=水平和垂直翻转）
aug_translate_enabled = os.getenv("AUG_TRANSLATE_ON", "0") != "0"                               #平移开关（0=不平移；1=平移；当启用时使用 AUG_TRANSLATE 指定的平移幅度进行随机平移）
aug_translate = float(os.getenv("AUG_TRANSLATE", "0.05"))                                       #平移幅度（相对于图象尺寸的比例，如 0.08 表示最多平移 8% 的图象宽度或高度；0 表示不平移）
aug_rotate_enabled = os.getenv("AUG_ROTATE_ON", "1") != "0"                                     #旋转开关（0=不旋转；1=旋转；当启用时使用 AUG_ROTATE 指定的旋转幅度进行随机旋转）
aug_rotate = float(os.getenv("AUG_ROTATE", "0.05"))                                             #旋转幅度（以 1.0 表示 360 度，如 0.08 表示最多旋转 28.8 度；0 表示不旋转）
aug_zoom_enabled = os.getenv("AUG_ZOOM_ON", "1") != "0"                                         #缩放开关（0=不缩放；1=缩放；当启用时使用 AUG_ZOOM 指定的缩放幅度进行随机缩放）
aug_zoom = float(os.getenv("AUG_ZOOM", "0.15"))                                                 #缩放幅度（相对于图象尺寸的比例，如 0.10 表示最多缩放 10% 的图象尺寸；0 表示不缩放）
aug_contrast_enabled = os.getenv("AUG_CONTRAST_ON", "1") != "0"                                 #对比度开关（0=不调整；1=调整；当启用时使用 AUG_CONTRAST_LOWER 和 AUG_CONTRAST_UPPER 指定的对比度范围进行随机调整）
aug_contrast_lower = float(os.getenv("AUG_CONTRAST_LOWER", "0.7"))                              #对比度下限（如 0.8 表示对比度最少为原来的 80%；1.0 表示不变；>1.0 表示增强；0 表示不调整对比度）
aug_contrast_upper = float(os.getenv("AUG_CONTRAST_UPPER", "1.3"))                              #对比度上限（如 1.2 表示对比度最多为原来的 120%；1.0 表示不变；<1.0 表示减弱；0 表示不调整对比度）
aug_brightness_enabled = os.getenv("AUG_BRIGHTNESS_ON", "1") != "0"                             #亮度开关（0=不调整；1=调整；当启用时使用 AUG_BRIGHTNESS_DELTA 指定的亮度调整幅度进行随机调整）
aug_brightness_delta = float(os.getenv("AUG_BRIGHTNESS_DELTA", "20.0"))                         #亮度调整幅度（如 25.5 表示亮度最多调整 ±10%；0 表示不调整亮度）
aug_saturation_enabled = os.getenv("AUG_SATURATION_ON", "1") != "0"                             #饱和度开关（0=不调整；1=调整；当启用时使用 AUG_SATURATION_LOWER 和 AUG_SATURATION_UPPER 指定的饱和度范围进行随机调整）
aug_saturation_lower = float(os.getenv("AUG_SATURATION_LOWER", "0.85"))                         #饱和度下限（如 0.85 表示饱和度最少为原来的 85%；1.0 表示不变；>1.0 表示增强；0 表示不调整饱和度）
aug_saturation_upper = float(os.getenv("AUG_SATURATION_UPPER", "1.15"))                         #饱和度上限（如 1.15 表示饱和度最多为原来的 115%；1.0 表示不变；<1.0 表示减弱；0 表示不调整饱和度）
aug_hue_enabled = os.getenv("AUG_HUE_ON", "1") != "0"                                           #色调开关（0=不调整；1=调整；当启用时使用 AUG_HUE_DELTA 指定的色调调整幅度进行随机调整）
aug_hue_delta = float(os.getenv("AUG_HUE_DELTA", "0.08"))                                       #色调调整幅度（以 1.0 表示 360 度，如 0.08 表示最多调整 ±28.8 度；0 表示不调整色调）
aug_gaussian_blur_enabled = os.getenv("AUG_GAUSSIAN_BLUR_ON", "1") != "0"                       #高斯模糊开关（0=不使用；1=使用；当启用时使用 AUG_GAUSSIAN_BLUR_KERNEL、AUG_GAUSSIAN_BLUR_SIGMA 和 AUG_GAUSSIAN_BLUR_PROB 指定的模糊核大小、标准差和应用概率进行随机高斯模糊）
aug_gaussian_blur_prob = float(os.getenv("AUG_GAUSSIAN_BLUR_PROB", "0.1"))                      #高斯模糊应用概率（如 0.2 表示每个图象有 20% 的概率应用高斯模糊；0 表示不使用高斯模糊）
aug_gaussian_blur_kernel = int(os.getenv("AUG_GAUSSIAN_BLUR_KERNEL", "3"))                      #高斯模糊核大小（会自动修正为 >=3 的奇数，且不超过图象最短边；如输入 4 会修正为 3）
aug_gaussian_blur_sigma = float(os.getenv("AUG_GAUSSIAN_BLUR_SIGMA", "0.0"))                    #高斯模糊标准差（如 0.0 表示自动根据核大小计算；>0 表示使用指定的标准差进行高斯模糊；当 AUG_GAUSSIAN_BLUR_ON=1 且 AUG_GAUSSIAN_BLUR_SIGMA=0 时将根据核大小自动计算标准差，通常为 kernel_size / 6）
aug_gaussian_noise_enabled = os.getenv("AUG_GAUSSIAN_NOISE_ON", "1") != "0"                     #高斯噪声开关（0=不添加；1=添加；当启用时使用 AUG_GAUSSIAN_NOISE 指定的标准差添加高斯噪声）
aug_gaussian_noise_stddev = float(os.getenv("AUG_GAUSSIAN_NOISE", "2"))                         #高斯噪声标准差（单位为像素值，当前图象范围为 0~255；例如 25.5 ≈ 255 的 10%；0 表示不添加高斯噪声）
aug_cutout_enabled = os.getenv("AUG_CUTOUT_ON", "0") != "0"                                     #Cutout 开关（0=不使用；1=使用；当启用时使用 AUG_CUTOUT 和 AUG_CUTOUT_PROB 指定区域大小与概率进行随机遮挡）
aug_cutout_ratio = float(os.getenv("AUG_CUTOUT", "0.0"))                                        #Cutout 区域大小（相对于图象尺寸的比例，如 0.2 表示最多遮挡 20% 的图象宽度和高度；0 表示不使用 Cutout）
aug_cutout_prob = float(os.getenv("AUG_CUTOUT_PROB", "0.5"))                                    #Cutout 应用概率（如 0.5 表示每个图象有 50% 的概率应用 Cutout；0 表示不使用 Cutout）
aug_grayscale_enabled = os.getenv("AUG_GRAYSCALE_ON", "0") != "0"                               #灰度化开关（0=不使用；1=使用；当启用时使用 AUG_GRAYSCALE_PROB 指定的灰度化概率进行随机灰度化）
aug_grayscale_prob = float(os.getenv("AUG_GRAYSCALE_PROB", "0.0"))                              #灰度化概率（如 0.3 表示每个图象有 30% 的概率转换为灰度图；0 表示不使用灰度化）
aug_motion_blur_enabled = os.getenv("AUG_MOTION_BLUR", "1") != "0"                              #运动模糊开关（0=不使用；1=使用；当启用时使用 AUG_MOTION_BLUR_KERNEL 和 AUG_MOTION_BLUR_PROB 指定的运动模糊核大小和应用概率进行随机运动模糊）
aug_motion_blur_prob = float(os.getenv("AUG_MOTION_BLUR_PROB", "0.3"))                          #运动模糊应用概率（如 0.3 表示每个图象有 30% 的概率应用运动模糊；0 表示不使用运动模糊）
aug_motion_blur_kernel = int(os.getenv("AUG_MOTION_BLUR_KERNEL", "3"))                          #运动模糊核大小（会自动修正为 >=3 的奇数，且不超过图象最短边；如输入 8 会修正为 7）
aug_motion_blur_dir = os.getenv("AUG_MOTION_BLUR_DIR", "random").strip().lower()                #运动模糊方向（random=随机水平或垂直；horizontal=水平；vertical=垂直；diag_*：模拟斜向运动）

# ================== 导出（EXPORT） ================== #
export_model_enabled = os.getenv("EXPORT_MODEL", "1") != "0"                                    #是否导出 .h5（0=不导出；1=导出）
export_tflite_enabled = os.getenv("EXPORT_TFLITE", "1") != "0"                                  #是否导出 .tflite（0=不导出；1=导出）
export_tflite_quantize = os.getenv("EXPORT_TFLITE_QUANTIZE", "none").strip().lower()       #TFLite 量化方式（none=不量化 float32；int8_dynamic=动态范围量化 权重int8 不兼容TFLite Micro；int8_full=全整数量化 权重+激活int8 兼容TFLite Micro；float16=半精度浮点）

# ================== 类别权重（CLASS_WEIGHT） ================== #
ckpt_verbose = int(os.getenv("CKPT_VERBOSE", "0"))                                              #checkpoint 的输出日志开关（0=安静；1=显示每个 epoch 保存提示）

class_weight_enabled = os.getenv("CLASS_WEIGHT", "1") != "0"                                    #类别权重开关（0=关闭；1=启用；当启用时将根据训练集类别分布自动计算权重，或使用 CLASS_WEIGHT_VALUES 指定的权重值；启用类别权重后训练时每个样本将乘以对应类别的权重，有助于缓解类别不平衡问题）
class_weight_values = os.getenv("CLASS_WEIGHT_VALUES")                                          #类别权重值（逗号分隔，如 "1.0,2.0,1.5"；当 CLASS_WEIGHT=1 且未设置 CLASS_WEIGHT_VALUES 时将根据训练集分布自动计算类别权重）
class_weight_mode = os.getenv("CLASS_WEIGHT_MODE", "dynamic").strip().lower()                   #类别权重模式（static=静态权重；dynamic=动态权重，根据每个 epoch 的分类指标自动调整类别权重；dynamic 需要配合 DYN_CW_* 参数进行调整策略设置）
dyn_class_weight_metric = os.getenv("DYN_CW_METRIC", "precision").strip().lower()               #动态类别权重监控指标（precision/recall/f1；根据每个 epoch 的该指标调整类别权重；建议优先使用验证指标，如 val_recall）
dyn_class_weight_beta = float(os.getenv("DYN_CW_BETA", "0.5"))                                  #动态类别权重平滑系数（0~1；新权重 = (1-beta)*当前权重 + beta*目标权重；beta 越大调整越激进，beta 越小调整越平滑）
dyn_class_weight_gamma = float(os.getenv("DYN_CW_GAMMA", "1.0"))                                #动态类别权重调整强度（>0；目标权重 = 基础权重 * (平均指标 / 每类指标)^gamma；gamma 越大调整越激烈，gamma 越小调整越温和；当 gamma=0 时不根据指标调整权重）
dyn_class_weight_min = float(os.getenv("DYN_CW_MIN", "0.2"))                                    #动态类别权重下限（>0；调整后的权重将被限制在 [DYN_CW_MIN, DYN_CW_MAX] 范围内，以避免过低或过高的权重导致训练不稳定）
dyn_class_weight_max = float(os.getenv("DYN_CW_MAX", "5.0"))                                    #动态类别权重上限（>0；调整后的权重将被限制在 [DYN_CW_MIN, DYN_CW_MAX] 范围内，以避免过低或过高的权重导致训练不稳定）
dyn_class_weight_verbose = os.getenv("DYN_CW_VERBOSE", "1") != "0"                              #动态类别权重调整日志开关（0=关闭；1=显示每个 epoch 的权重调整提示）

# ================== 训练评估（EVAL_TRAIN） ================== #
eval_train_enabled = os.getenv("EVAL_TRAIN", "1") != "0"                                        #训练评估开关（0=关闭；1=启用；当启用时每隔 EVAL_TRAIN_FREQ 个 epoch 或在训练结束时评估一次训练集，评估结果将以 train_eval_* 的形式添加到日志中，并在控制台输出 train_eval_loss 和 train_eval_accuracy）
eval_train_freq = int(os.getenv("EVAL_TRAIN_FREQ", "5"))                                        #训练评估频率（多少个 epoch 评估一次训练集；当 EVAL_TRAIN=1 且 EVAL_TRAIN_FREQ=0 时表示只在训练结束时评估一次；当 EVAL_TRAIN=1 且 EVAL_TRAIN_FREQ>0 时表示每隔多少个 epoch 评估一次训练集）
eval_train_steps = int(os.getenv("EVAL_TRAIN_STEPS", "0"))                                      #训练评估最大步数（每次评估时使用训练集的前多少个 batch；0 表示使用全部训练数据；当训练集较大时可以设置为较小的值以加快评估速度）

# ================== 固定路径（通常无需修改） ================== #
latest_checkpoint_model_path = os.path.join(checkpoint_dir, "latest_model.h5")
best_checkpoint_model_path = os.path.join(checkpoint_dir, "best_model.h5")
best_metric_path = os.path.join(checkpoint_dir, "best_metric.json")
train_state_path = os.path.join(checkpoint_dir, "train_state.json")

def _load_train_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"epoch": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        epoch = int(data.get("epoch", 0))
        return {"epoch": max(0, epoch)}
    except Exception:
        return {"epoch": 0}


def _save_train_state(path: str, epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"epoch": int(epoch)}, f, ensure_ascii=False)


def _load_best_metric(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_best_metric(path: str, monitor: str, value: float, epoch: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"monitor": str(monitor), "value": float(value), "epoch": int(epoch)},
            f,
            ensure_ascii=False
        )


def _infer_mode(monitor: str) -> str:
    name = (monitor or "").lower()
    return "min" if "loss" in name else "max"


class BestModelSaver(tf.keras.callbacks.Callback):
    def __init__(
        self,
        filepath: str,
        metric_path: str,
        monitor: str = "val_accuracy",
        fallback_monitor: str = "accuracy",
        mode: str | None = None,
        fallback_mode: str | None = None
    ) -> None:
        super().__init__()
        self.filepath = filepath
        self.metric_path = metric_path
        self.monitor = monitor
        self.fallback_monitor = fallback_monitor
        self.mode = mode or _infer_mode(monitor)
        self.fallback_mode = fallback_mode or _infer_mode(fallback_monitor)
        self.best = None
        self.best_monitor = None

    def on_train_begin(self, logs=None):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        state = _load_best_metric(self.metric_path)
        monitor = state.get("monitor")
        value = state.get("value")
        if isinstance(monitor, str) and isinstance(value, (int, float)):
            self.best_monitor = monitor
            self.best = float(value)

    def _is_improved(self, value: float, mode: str) -> bool:
        if self.best is None:
            return True
        if mode == "min":
            return value < self.best
        return value > self.best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        monitor = self.monitor
        value = logs.get(monitor)
        mode = self.mode

        if value is None and self.fallback_monitor and best_allow_fallback:
            monitor = self.fallback_monitor
            value = logs.get(monitor)
            mode = self.fallback_mode

        if value is None:
            return

        value = float(value)

        if self.best_monitor is not None and self.best_monitor != monitor:
            self.best = None
            self.best_monitor = monitor

        if not self._is_improved(value, mode):
            return

        self.best = value
        self.best_monitor = monitor
        self.model.save(self.filepath)
        _save_best_metric(self.metric_path, monitor, value, int(epoch) + 1)
        if best_verbose:
            print(f"Epoch {int(epoch) + 1}: {monitor} improved to {value:.6f}, saving best model to {self.filepath}")


def _compute_classification_metrics(model: tf.keras.Model, dataset, num_classes: int) -> dict:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch_x, batch_y in dataset:
        y_true = tf.argmax(batch_y, axis=1)
        y_pred = tf.argmax(model(batch_x, training=False), axis=1)
        confusion += tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    support = confusion.sum(axis=1).astype(np.int64)

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "confusion": confusion,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


class ClassificationMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, class_names: list[str], num_classes: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.class_names = class_names
        self.num_classes = num_classes
        self.last_metrics = None
        self.last_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if not any(k.startswith("val_") for k in logs.keys()):
            return

        metrics = _compute_classification_metrics(self.model, self.dataset, self.num_classes)
        self.last_metrics = metrics
        self.last_epoch = int(epoch)
        logs["val_precision"] = metrics["macro_precision"]
        logs["val_recall"] = metrics["macro_recall"]
        logs["val_f1"] = metrics["macro_f1"]

        print(
            f"\nEpoch {int(epoch) + 1}: val_precision={metrics['macro_precision']:.4f} "
            f"val_recall={metrics['macro_recall']:.4f} val_f1={metrics['macro_f1']:.4f}"
        )
        for i in range(self.num_classes):
            name = self.class_names[i] if i < len(self.class_names) else str(i)
            print(
                f"  {name}: precision={metrics['precision'][i]:.4f} "
                f"recall={metrics['recall'][i]:.4f} f1={metrics['f1'][i]:.4f} "
                f"support={int(metrics['support'][i])}"
            )


class DynamicClassWeightCallback(tf.keras.callbacks.Callback):
    def __init__(self, weight_vec: tf.Variable, metrics_callback: ClassificationMetricsCallback) -> None:
        super().__init__()
        self.weight_vec = weight_vec
        self.metrics_callback = metrics_callback
        self.base_weight = None

    def on_train_begin(self, logs=None):
        self.base_weight = self.weight_vec.numpy().astype(np.float32)

    def on_epoch_end(self, epoch, logs=None):
        metrics = getattr(self.metrics_callback, "last_metrics", None)
        if metrics is None or getattr(self.metrics_callback, "last_epoch", None) != int(epoch):
            return

        name = dyn_class_weight_metric
        if name == "precision":
            per_class = metrics["precision"]
        elif name == "f1":
            per_class = metrics["f1"]
        else:
            per_class = metrics["recall"]

        per_class = np.asarray(per_class, dtype=np.float32)
        eps = 1e-7
        target = float(np.mean(per_class))
        factor = np.power(target / (per_class + eps), dyn_class_weight_gamma)

        base = self.base_weight if self.base_weight is not None else self.weight_vec.numpy().astype(np.float32)
        proposed = base * factor
        proposed = proposed / (float(np.mean(proposed)) + eps)
        proposed = np.clip(proposed, dyn_class_weight_min, dyn_class_weight_max).astype(np.float32)

        beta = float(np.clip(dyn_class_weight_beta, 0.0, 1.0))
        current = self.weight_vec.numpy().astype(np.float32)
        updated = (1.0 - beta) * current + beta * proposed
        self.weight_vec.assign(updated)

        if dyn_class_weight_verbose:
            as_list = [float(x) for x in self.weight_vec.numpy().tolist()]
            print(f"Epoch {int(epoch) + 1}: class_weight updated ({name}) -> {as_list}")


class OnlyWhenMetricAvailable(tf.keras.callbacks.Callback):
    def __init__(self, callback: tf.keras.callbacks.Callback, monitor: str) -> None:
        super().__init__()
        self.callback = callback
        self.monitor = monitor

    def set_model(self, model):
        super().set_model(model)
        self.callback.set_model(model)

    def set_params(self, params):
        super().set_params(params)
        self.callback.set_params(params)

    def on_train_begin(self, logs=None):
        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.monitor in logs:
            self.callback.on_epoch_end(epoch, logs=logs)


def _build_lr_scheduler_callback() -> tf.keras.callbacks.Callback | None:
    if lr_sched in {"none", "off", "false", "0"}:
        return None

    if lr_sched in {"plateau", "reduce", "reducelronplateau"}:
        cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=lr_plateau_monitor,
            factor=lr_plateau_factor,
            patience=lr_plateau_patience,
            min_lr=lr_plateau_min_lr,
            cooldown=lr_plateau_cooldown,
            verbose=1
        )
        return OnlyWhenMetricAvailable(cb, lr_plateau_monitor)

    if lr_sched in {"cosine", "cos"}:
        def schedule(epoch, lr):
            total = max(1, epochs)
            t = min(float(epoch) / float(total), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * t))
            return float(lr_cosine_min_lr + (learning_rate - lr_cosine_min_lr) * cosine)

        return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)

    return None


class TrainSetEvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, every_n_epochs: int = 0, max_steps: int = 0) -> None:
        super().__init__()
        self.dataset = dataset
        self.every_n_epochs = int(max(0, every_n_epochs))
        self.max_steps = int(max(0, max_steps))

    def _evaluate(self, logs, epoch: int) -> None:
        ds = self.dataset
        if self.max_steps > 0:
            ds = ds.take(self.max_steps)
        results = self.model.evaluate(ds, verbose=0, return_dict=True)
        for k, v in results.items():
            logs[f"train_eval_{k}"] = float(v)
        if "loss" in results and "accuracy" in results:
            print(
                f"\nEpoch {int(epoch) + 1}: train_eval_loss={float(results['loss']):.4f} "
                f"train_eval_accuracy={float(results['accuracy']):.4f}"
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.every_n_epochs <= 0:
            return
        if (int(epoch) + 1) % self.every_n_epochs != 0:
            return
        self._evaluate(logs, int(epoch))

    def on_train_end(self, logs=None):
        if self.every_n_epochs > 0:
            return
        logs = logs or {}
        self._evaluate(logs, epoch=-1)


def _compute_class_weight_from_dataset(dataset, num_classes: int) -> tuple[dict, np.ndarray]:
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, batch_y in dataset:
        idx = tf.argmax(batch_y, axis=1, output_type=tf.int32)
        bincount = tf.math.bincount(idx, minlength=num_classes, maxlength=num_classes).numpy()
        counts += bincount.astype(np.int64)

    total = int(counts.sum())
    weights = {}
    for i in range(num_classes):
        if counts[i] <= 0:
            weights[i] = 0.0
        else:
            weights[i] = float(total / (num_classes * int(counts[i])))
    return weights, counts


def _apply_class_weight_to_dataset(dataset, class_weight: dict) -> tf.data.Dataset:
    return _apply_class_weight_vec_to_dataset(
        dataset,
        tf.constant([float(class_weight[i]) for i in range(num_classes)], dtype=tf.float32)
    )


def _apply_class_weight_vec_to_dataset(dataset, weight_vec) -> tf.data.Dataset:
    def add_sample_weight(batch_x, batch_y):
        idx = tf.argmax(batch_y, axis=1, output_type=tf.int32)
        sw = tf.gather(weight_vec, idx)
        return batch_x, batch_y, sw

    return dataset.map(add_sample_weight, num_parallel_calls=tf.data.AUTOTUNE)

# ========== 创建训练/验证数据集 ==========
if os.path.abspath(train_data_dir) == os.path.abspath(validation_data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        validation_split=validation_split,
        subset="training",
        color_mode='rgb',
        label_mode='categorical',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_data_dir,
        validation_split=validation_split,
        subset="validation",
        color_mode='rgb',
        label_mode='categorical',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
else:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        color_mode='rgb',
        label_mode='categorical',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_data_dir,
        color_mode='rgb',
        label_mode='categorical',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

train_eval_ds = train_ds

class_names = train_ds.class_names

autotune = tf.data.AUTOTUNE

augmentation_layers = []
flip_mode = aug_flip_mode
if flip_mode in {"both", "hv", "h_v", "horizontal_vertical", "horizontal+vertical"}:
    flip_mode = "horizontal_and_vertical"
if aug_flip_enabled and flip_mode not in {"0", "false", "none", "off"}:
    augmentation_layers.append(layers.RandomFlip(flip_mode))
if aug_translate_enabled and aug_translate > 0:
    augmentation_layers.append(layers.RandomTranslation(aug_translate, aug_translate, fill_mode="reflect"))
if aug_rotate_enabled and aug_rotate > 0:
    augmentation_layers.append(layers.RandomRotation(aug_rotate, fill_mode="reflect"))
if aug_zoom_enabled and aug_zoom > 0:
    augmentation_layers.append(layers.RandomZoom(aug_zoom, aug_zoom, fill_mode="reflect"))

data_augmentation = tf.keras.Sequential(augmentation_layers, name="data_augmentation")


def _random_cutout(images: tf.Tensor, ratio: float, prob: float) -> tf.Tensor:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    prob = float(np.clip(prob, 0.0, 1.0))
    if ratio <= 0.0 or prob <= 0.0:
        return images

    shape = tf.shape(images)
    b = shape[0]
    h = shape[1]
    w = shape[2]

    cut_h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    cut_w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
    cut_h = tf.maximum(1, cut_h)
    cut_w = tf.maximum(1, cut_w)

    cy = tf.random.uniform([b], minval=0, maxval=h, dtype=tf.int32)
    cx = tf.random.uniform([b], minval=0, maxval=w, dtype=tf.int32)

    top = tf.clip_by_value(cy - cut_h // 2, 0, h)
    bottom = tf.clip_by_value(top + cut_h, 0, h)
    left = tf.clip_by_value(cx - cut_w // 2, 0, w)
    right = tf.clip_by_value(left + cut_w, 0, w)

    y = tf.range(h, dtype=tf.int32)[None, :, None]
    x = tf.range(w, dtype=tf.int32)[None, None, :]
    y_in = tf.logical_and(y >= top[:, None, None], y < bottom[:, None, None])
    x_in = tf.logical_and(x >= left[:, None, None], x < right[:, None, None])
    region = tf.logical_and(y_in, x_in)

    apply = tf.random.uniform([b], 0.0, 1.0) < prob
    region = tf.logical_and(region, apply[:, None, None])
    region = region[..., None]

    fill = tf.zeros([], dtype=images.dtype)
    return tf.where(region, fill, images)


def _random_gaussian_blur(images: tf.Tensor, kernel_size: int, sigma: float, prob: float) -> tf.Tensor:
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return images

    shape = tf.shape(images)
    h = shape[1]
    w = shape[2]
    c = shape[3]

    k = tf.cast(kernel_size, tf.int32)
    k = tf.maximum(3, k)
    k = tf.minimum(k, tf.minimum(h, w))
    k = tf.maximum(3, k)
    k = tf.where(tf.equal(k % 2, 0), k - 1, k)
    k = tf.maximum(3, k)

    sigma_val = float(sigma)
    if sigma_val <= 0.0:
        sigma_val = 0.3 * (((float(kernel_size) - 1.0) * 0.5) - 1.0) + 0.8
        sigma_val = max(0.3, sigma_val)

    dtype = images.dtype
    center = k // 2
    coords = tf.cast(tf.range(k) - center, tf.float32)
    denom = 2.0 * (sigma_val ** 2)
    g = tf.exp(-(coords ** 2) / denom)
    g = g / (tf.reduce_sum(g) + 1e-7)
    kernel_2d = tf.tensordot(g, g, axes=0)
    kernel_2d = tf.cast(kernel_2d, dtype)
    kernel_2d = kernel_2d / (tf.reduce_sum(kernel_2d) + tf.cast(1e-7, dtype))
    kernel = kernel_2d[:, :, None, None]
    kernel = tf.tile(kernel, [1, 1, c, 1])

    blurred = tf.nn.depthwise_conv2d(images, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
    p = tf.random.uniform([shape[0], 1, 1, 1], 0.0, 1.0)
    return tf.where(p < prob, blurred, images)


def _random_motion_blur(images: tf.Tensor, kernel_size: int, prob: float) -> tf.Tensor:
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= 0.0:
        return images

    shape = tf.shape(images)
    h = shape[1]
    w = shape[2]
    c = shape[3]

    k = tf.cast(kernel_size, tf.int32)
    k = tf.maximum(3, k)
    k = tf.minimum(k, tf.minimum(h, w))
    k = tf.maximum(3, k)
    k = tf.where(tf.equal(k % 2, 0), k - 1, k)
    k = tf.maximum(3, k)

    dir_raw = aug_motion_blur_dir
    dir_map = {
        "h": 0,
        "horizontal": 0,
        "x": 0,
        "v": 1,
        "vertical": 1,
        "y": 1,
        "diag": 2,
        "diag_down": 2,
        "down": 2,
        "d": 2,
        "diag_up": 3,
        "up": 3,
        "u": 3,
        "random": -1,
        "all": -1,
        "*": -1,
    }
    if dir_raw in dir_map and dir_map[dir_raw] >= 0:
        dirs = [dir_map[dir_raw]]
    else:
        tokens = [t for t in dir_raw.replace("|", ",").replace(" ", ",").split(",") if t]
        dirs = [dir_map[t] for t in tokens if t in dir_map and dir_map[t] >= 0]
        if not dirs:
            dirs = [0, 1, 2, 3]

    dirs_tensor = tf.constant(dirs, dtype=tf.int32)
    choice = tf.random.uniform([], minval=0, maxval=tf.size(dirs_tensor), dtype=tf.int32)
    direction = dirs_tensor[choice]
    center = k // 2
    eye = tf.eye(k, dtype=images.dtype)
    horizontal = tf.one_hot(center, k, dtype=images.dtype)[None, :]
    horizontal = tf.tile(horizontal, [k, 1])
    vertical = tf.transpose(horizontal)
    diag_down = eye
    diag_up = tf.reverse(eye, axis=[1])
    kernel_2d = tf.switch_case(
        direction,
        branch_fns=[
            lambda: horizontal,
            lambda: vertical,
            lambda: diag_down,
            lambda: diag_up,
        ],
        default=lambda: horizontal,
    )
    kernel_2d = kernel_2d / (tf.reduce_sum(kernel_2d) + tf.cast(1e-7, images.dtype))
    kernel = kernel_2d[:, :, None, None]
    kernel = tf.tile(kernel, [1, 1, c, 1])

    def apply_blur():
        return tf.nn.depthwise_conv2d(images, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")

    return tf.cond(tf.random.uniform([], 0.0, 1.0) < prob, apply_blur, lambda: images)


def _augment(images, labels):
    images = data_augmentation(images, training=True)
    if aug_motion_blur_enabled:
        images = _random_motion_blur(images, kernel_size=aug_motion_blur_kernel, prob=aug_motion_blur_prob)
    if aug_gaussian_blur_enabled:
        images = _random_gaussian_blur(
            images,
            kernel_size=aug_gaussian_blur_kernel,
            sigma=aug_gaussian_blur_sigma,
            prob=aug_gaussian_blur_prob,
        )
    if aug_contrast_enabled and aug_contrast_lower > 0 and aug_contrast_upper > 0:
        images = tf.image.random_contrast(images, lower=aug_contrast_lower, upper=aug_contrast_upper)
    if aug_brightness_enabled and aug_brightness_delta > 0:
        images = tf.image.random_brightness(images, max_delta=aug_brightness_delta)
    if img_depth == 3:
        if aug_saturation_enabled and aug_saturation_lower > 0 and aug_saturation_upper > 0:
            images = tf.image.random_saturation(images, lower=aug_saturation_lower, upper=aug_saturation_upper)
        if aug_hue_enabled and aug_hue_delta > 0:
            images = tf.image.random_hue(images, max_delta=aug_hue_delta)
        if aug_grayscale_enabled and aug_grayscale_prob > 0:
            p = tf.random.uniform([tf.shape(images)[0], 1, 1, 1], 0.0, 1.0)
            gray = tf.image.rgb_to_grayscale(images)
            gray = tf.tile(gray, [1, 1, 1, 3])
            images = tf.where(p < aug_grayscale_prob, gray, images)
    if aug_gaussian_noise_enabled and aug_gaussian_noise_stddev > 0:
        images = images + tf.random.normal(tf.shape(images), mean=0.0, stddev=aug_gaussian_noise_stddev, dtype=images.dtype)
    if aug_cutout_enabled and aug_cutout_ratio > 0:
        images = _random_cutout(images, ratio=aug_cutout_ratio, prob=aug_cutout_prob)
    images = tf.clip_by_value(images, 0.0, 255.0)
    return images, labels


if augment_enabled:
    train_ds = train_ds.map(_augment, num_parallel_calls=autotune)

dynamic_class_weight_vec = None
if class_weight_enabled:
    if class_weight_values:
        values = [v.strip() for v in class_weight_values.split(",") if v.strip()]
        if len(values) == num_classes:
            class_weight = {i: float(values[i]) for i in range(num_classes)}
        else:
            class_weight = None
    else:
        class_weight, counts = _compute_class_weight_from_dataset(train_ds, num_classes)
        print("Class counts:", counts.tolist())

    if class_weight is not None:
        print("Class weight:", class_weight)
        if class_weight_mode in {"dynamic", "dyn"}:
            dynamic_class_weight_vec = tf.Variable(
                [float(class_weight[i]) for i in range(num_classes)],
                dtype=tf.float32,
                trainable=False
            )
            train_ds = _apply_class_weight_vec_to_dataset(train_ds, dynamic_class_weight_vec)
        else:
            train_ds = _apply_class_weight_to_dataset(train_ds, class_weight)

train_ds = train_ds.prefetch(autotune)
val_ds = val_ds.prefetch(autotune)
train_eval_ds = train_eval_ds.prefetch(autotune)

# 获取类别名称
print("Class names:", class_names)
samplext = next(iter(train_ds))
xx = samplext[0]
print(xx.shape)

# 设置模型层级
def _build_model() -> tf.keras.Model:
    model = models.Sequential([
        layers.InputLayer(input_shape=(img_height, img_width, img_depth)),
        layers.Conv2D(8, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),                                # 32→16
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),                                # 16→8
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),                                # 8→4
        layers.GlobalAveragePooling2D(),                          # 32维，替代Flatten
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


os.makedirs(checkpoint_dir, exist_ok=True)
train_state = _load_train_state(train_state_path)
initial_epoch = int(train_state.get("epoch", 0))

model_path_to_load = None
if os.path.exists(latest_checkpoint_model_path):
    model_path_to_load = latest_checkpoint_model_path
elif os.path.exists(Model_name):
    model_path_to_load = Model_name

if model_path_to_load is not None:
    model = tf.keras.models.load_model(model_path_to_load)
else:
    model = _build_model()

model.summary()

# 模型编译
if getattr(model, "optimizer", None) is None:
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                   loss=losses.CategoricalCrossentropy(from_logits=False),
                   metrics=['accuracy'])

# 模型训练
metrics_cb = ClassificationMetricsCallback(
    dataset=val_ds,
    class_names=class_names,
    num_classes=num_classes
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=latest_checkpoint_model_path,
        save_best_only=False,
        save_weights_only=False,
        verbose=ckpt_verbose
    ),
    BestModelSaver(
        filepath=best_checkpoint_model_path,
        metric_path=best_metric_path,
        monitor=best_monitor,
        fallback_monitor=best_fallback_monitor
    ),
    metrics_cb,
    TrainSetEvalCallback(train_eval_ds, every_n_epochs=eval_train_freq, max_steps=eval_train_steps) if eval_train_enabled else None,
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: _save_train_state(train_state_path, int(epoch) + 1)
    )
]
callbacks = [cb for cb in callbacks if cb is not None]

if dynamic_class_weight_vec is not None:
    callbacks.append(DynamicClassWeightCallback(dynamic_class_weight_vec, metrics_cb))

if early_stop_enabled:
    callbacks.append(
        OnlyWhenMetricAvailable(
            tf.keras.callbacks.EarlyStopping(
            monitor=early_stop_monitor,
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
            mode=early_stop_mode,
            restore_best_weights=early_stop_restore_best_weights,
            verbose=1
            ),
            early_stop_monitor
        )
    )

lr_scheduler_callback = _build_lr_scheduler_callback()
if lr_scheduler_callback is not None:
    callbacks.append(lr_scheduler_callback)

history = None
try:
    if initial_epoch < epochs:
        history = model.fit(
            train_ds,
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=batch_size,
            validation_data=val_ds,
            validation_freq=validation_freq,
            callbacks=callbacks
        )
        _save_train_state(train_state_path, epochs)
except KeyboardInterrupt:
    print("\n训练被中断：将按最优结果导出模型文件。")

# 保存模型文件
final_model_path = None
if os.path.exists(best_checkpoint_model_path):
    final_model_path = best_checkpoint_model_path
elif os.path.exists(latest_checkpoint_model_path):
    final_model_path = latest_checkpoint_model_path

if final_model_path is not None:
    final_model = tf.keras.models.load_model(final_model_path)
else:
    final_model = model

final_metrics = _compute_classification_metrics(final_model, val_ds, num_classes)
print(
    f"最终模型指标: precision={final_metrics['macro_precision']:.4f} "
    f"recall={final_metrics['macro_recall']:.4f} f1={final_metrics['macro_f1']:.4f}"
)
for i in range(num_classes):
    name = class_names[i] if i < len(class_names) else str(i)
    print(
        f"  {name}: precision={final_metrics['precision'][i]:.4f} "
        f"recall={final_metrics['recall'][i]:.4f} f1={final_metrics['f1'][i]:.4f} "
        f"support={int(final_metrics['support'][i])}"
    )

if export_model_enabled:
    final_model.save(Model_name)

if export_tflite_enabled:
    converter = tf.lite.TFLiteConverter.from_keras_model(final_model)

    if export_tflite_quantize in {"int8_dynamic", "int8", "dynamic", "drq"}:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quant_type = "int8 (动态范围量化, 权重int8/激活float32)"
    elif export_tflite_quantize in {"int8_full", "full", "full_int", "int8_full_integer"}:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def _representative_dataset():
            for batch_x, _ in val_ds.take(200):
                yield [batch_x]

        converter.representative_dataset = _representative_dataset
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        quant_type = "int8 (全整数量化, 权重int8/激活int8)"
    elif export_tflite_quantize in {"float16", "f16", "fp16", "half"}:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        quant_type = "float16 (半精度浮点)"
    else:
        quant_type = "float32 (无量化)"

    tflite_model = converter.convert()
    tflite_name = os.path.splitext(Model_name)[0] + ".tflite"
    with open(tflite_name, "wb") as f:
        f.write(tflite_model)
    print(f"成功生成TFLite文件！量化方式: {quant_type}")
