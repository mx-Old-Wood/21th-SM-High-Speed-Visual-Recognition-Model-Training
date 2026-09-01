# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 望着天空的眼睛 <a15234181830@163.com>
# Modified in 2026 for 21th SM High-Speed Visual Recognition.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import json
import math
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxsim import simplify
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


class MotionBlur:
    """Apply random motion blur with given kernel size."""
    def __init__(self, max_kernel_size=9, p=0.3):
        self.max_kernel_size = max_kernel_size
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        k = random.randrange(3, self.max_kernel_size + 1, 2)
        angle = random.uniform(0, 360)
        kernel = torch.zeros((k, k))
        center = k // 2
        rad = math.radians(angle)
        dx, dy = math.cos(rad), math.sin(rad)
        for i in range(k):
            t = i - center
            x = center + int(round(t * dx))
            y = center + int(round(t * dy))
            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] = 1.0
        kernel /= kernel.sum()
        kernel = kernel.expand(3, 1, k, k)
        img_t = F.pad(transforms.functional.to_tensor(img).unsqueeze(0), [center] * 4, mode="reflect")
        blurred = F.conv2d(img_t, kernel, groups=3)
        return transforms.ToPILImage()(blurred.squeeze(0))


class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy examples, up-weights hard ones."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        ce = nn.functional.cross_entropy(logits, labels, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DWConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 1.0):
        super().__init__()

        def c(ch: int) -> int:
            return max(8, int(ch * width_mult))

        self.features = nn.Sequential(
            nn.Conv2d(3, c(12), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(12)),
            nn.ReLU(inplace=True),
            DWConvBlock(c(12), c(16), stride=1),
            DWConvBlock(c(16), c(24), stride=2),
            DWConvBlock(c(24), c(32), stride=1),
            DWConvBlock(c(32), c(48), stride=2),
            DWConvBlock(c(48), c(48), stride=1),
            nn.Conv2d(c(48), c(64), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c(64)),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c(64), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_splits(image_folder: datasets.ImageFolder, seed: int, train_ratio: float, val_ratio: float):
    per_class_indices = {idx: [] for idx in range(len(image_folder.classes))}
    for sample_index, (_, class_idx) in enumerate(image_folder.samples):
        per_class_indices[class_idx].append(sample_index)

    train_indices = []
    val_indices = []
    test_indices = []

    rng = random.Random(seed)
    for _, indices in per_class_indices.items():
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def get_data_loaders(data_root: Path, img_size: int, batch_size: int, num_workers: int, seed: int, no_augment: bool = False):
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if no_augment:
        train_tf = eval_tf
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(8),
            transforms.RandomAffine(degrees=0, scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.03),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    base_dataset = datasets.ImageFolder(str(data_root), transform=None)
    train_indices, val_indices, test_indices = build_splits(base_dataset, seed, train_ratio=0.8, val_ratio=0.1)

    train_dataset = datasets.ImageFolder(str(data_root), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(data_root), transform=eval_tf)
    test_dataset = datasets.ImageFolder(str(data_root), transform=eval_tf)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, persistent_workers=True, prefetch_factor=4)
    return train_loader, val_loader, test_loader, base_dataset.classes, base_dataset.class_to_idx, base_dataset


def run_epoch(model, loader, criterion, optimizer, device, desc=""):
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    with torch.set_grad_enabled(train_mode):
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

            pbar.set_postfix(loss=f"{total_loss / max(1, total_count):.4f}",
                             acc=f"{total_correct / max(1, total_count):.4f}")

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc


def benchmark_torch_cpu(model: nn.Module, img_size: int, loops: int = 500, warmup: int = 50):
    model = model.cpu().eval()
    torch.set_num_threads(1)
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        start = time.perf_counter()
        for _ in range(loops):
            _ = model(x)
        end = time.perf_counter()
    avg_ms = (end - start) * 1000.0 / loops
    return avg_ms


def export_onnx(model: nn.Module, onnx_path: Path, img_size: int):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    onnx_model, check = simplify(str(onnx_path))
    if not check:
        raise RuntimeError("onnxsim simplify failed")
    import onnx
    onnx.save(onnx_model, str(onnx_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset",
        help="Dataset root folder (ImageFolder structure), e.g. dataset or D:/my_data/cls_dataset",
    )
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width-mult", type=float, default=0.6)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--target-acc", type=float, default=0.95)
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation (train with clean images only)")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"data-root not found or not a directory: {data_root}")

    class_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    if len(class_dirs) < 2:
        raise RuntimeError(f"data-root must contain at least 2 class folders: {data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    print(f"Data root: {data_root}")

    train_loader, val_loader, test_loader, classes, class_to_idx, base_dataset = get_data_loaders(
        data_root=data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        no_augment=args.no_augment,
    )

    model = TinyClassifier(num_classes=len(classes), width_mult=args.width_mult).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Classes: {classes}")
    print(f"Train/Val/Test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"Model params: {n_params}")

    # Per-class inverse-frequency weights for imbalanced training set
    train_class_counts = [0] * len(classes)
    for idx in train_loader.dataset.indices:
        _, label = base_dataset.samples[idx]
        train_class_counts[label] += 1
    class_weights = torch.tensor(
        [sum(train_class_counts) / (len(classes) * c) for c in train_class_counts],
        device=device,
    )
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = 3
    total_epochs = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    best_epoch = -1
    best_path = out_dir / "best_model.pt"
    bad_epochs = 0
    reached_target_acc = False

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device,
                                          desc=f"Epoch {epoch:02d}/{args.epochs} [train]")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device,
                                      desc=f"Epoch {epoch:02d}/{args.epochs} [val]")
        scheduler.step()

        epoch_time = time.time() - epoch_start
        eta = timedelta(seconds=int((args.epochs - epoch) * epoch_time))

        print(
            f"Epoch {epoch:02d}/{args.epochs} [{epoch_time:.1f}s, ETA {eta}] | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "class_to_idx": class_to_idx,
                    "img_size": args.img_size,
                    "width_mult": args.width_mult,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                best_path,
            )
        else:
            bad_epochs += 1

        if best_val_acc >= args.target_acc:
            reached_target_acc = True
            print(
                f"Reached target val_acc={args.target_acc:.4f} at epoch {epoch}, "
                f"best_val_acc={best_val_acc:.4f}. Stop training and continue next stage."
            )
            break

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}, best epoch = {best_epoch}")
            break

    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device)
    print(f"Best val_acc={best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Test loss={test_loss:.4f}, test acc={test_acc:.4f}")

    model_cpu = model.cpu().eval()
    onnx_path = out_dir / "tiny_classifier_fp32.onnx"
    export_onnx(model_cpu, onnx_path, args.img_size)

    cpu_ms = benchmark_torch_cpu(model_cpu, args.img_size)
    print(f"Torch CPU(1 thread) avg latency: {cpu_ms:.3f} ms")

    metrics = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "target_acc": args.target_acc,
        "reached_target_acc": reached_target_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "torch_cpu_1thread_ms": cpu_ms,
        "num_params": n_params,
        "classes": classes,
        "onnx_path": str(onnx_path),
        "img_size": args.img_size,
        "width_mult": args.width_mult,
    }

    labels_path = out_dir / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for name in classes:
            f.write(f"{name}\n")

    metrics["labels_path"] = str(labels_path)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved: {best_path}")
    print(f"Saved: {onnx_path}")
    print(f"Saved: {labels_path}")
    print(f"Saved: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
