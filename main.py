"""
Project 37: CutMix and MixUp Data Augmentation on CIFAR-10
==========================================================
Description:
Implementation and empirical comparative evaluation of CutMix (Yun et al., ICCV 2019)
and MixUp (Zhang et al., ICLR 2018) advanced data augmentation strategies on CIFAR-10
using a custom Deep Residual Network (MiniResNet). Demonstrates convex input blending,
regional pixel replacement with area-proportional label smoothing, overconfidence
mitigation, and generalization improvements.
"""

import copy
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# Configuration & Seed Initialization
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 5
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 3000
NUM_VAL_SAMPLES = 1000

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def set_seed(seed: int = 42) -> None:
    """Ensure strict deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# CutMix and MixUp Augmentation Primitives
# -----------------------------------------------------------------------------
def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp: convex linear combination of image pairs and target labels.
    x_tilde = lambda * x_i + (1 - lambda) * x_j
    """
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    """
    Computes a random bounding box coordinates matching the area proportion (1 - lam).
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform random center of bounding box
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies CutMix: cuts a patch from image j and pastes it onto image i.
    Adjusts lambda to the exact proportional area replaced.
    """
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exact replaced pixel area ratio
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return mixed_x, y_a, y_b, lam


def mixed_criterion(
    criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
) -> torch.Tensor:
    """Calculates weighted linear combination of cross-entropy losses."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# -----------------------------------------------------------------------------
# Deep Residual Architecture (MiniResNet)
# -----------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Standard pre-activation / post-activation residual block with projection shortcut."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class MiniResNet(nn.Module):
    """
    Compact 4-stage Residual Convolutional Network designed for rapid
    CIFAR-10 training and high-fidelity representation learning.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = ResidualBlock(32, 64, stride=2)   # 32x32 -> 16x16
        self.stage2 = ResidualBlock(64, 128, stride=2)  # 16x16 -> 8x8
        self.stage3 = ResidualBlock(128, 256, stride=2) # 8x8 -> 4x4

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.in_conv(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


# -----------------------------------------------------------------------------
# Dataset Preparation
# -----------------------------------------------------------------------------
def get_cifar10_dataloaders(
    data_dir: Path,
    num_train: int = NUM_TRAIN_SAMPLES,
    num_val: int = NUM_VAL_SAMPLES,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, datasets.CIFAR10]:
    """Prepares standard CIFAR-10 dataloaders with standard spatial normalization."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    full_train = datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=train_transform)
    full_test = datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=test_transform)

    train_subset = Subset(full_train, list(range(min(num_train, len(full_train)))))
    test_subset = Subset(full_test, list(range(min(num_val, len(full_test)))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, full_test


# -----------------------------------------------------------------------------
# Training and Evaluation Engine
# -----------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    regime: str,
    device: torch.device,
) -> Tuple[float, float]:
    """Trains for one epoch under the specified augmentation regime."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        if regime == "baseline":
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        elif regime == "mixup":
            mixed_x, y_a, y_b, lam = mixup_data(inputs, targets, alpha=0.8)
            outputs = model(mixed_x)
            loss = mixed_criterion(criterion, outputs, y_a, y_b, lam)
        elif regime == "cutmix":
            mixed_x, y_a, y_b, lam = cutmix_data(inputs, targets, alpha=1.0)
            outputs = model(mixed_x)
            loss = mixed_criterion(criterion, outputs, y_a, y_b, lam)
        elif regime == "cutmix_mixup":
            # 50% probability CutMix, 50% MixUp
            if random.random() < 0.5:
                mixed_x, y_a, y_b, lam = cutmix_data(inputs, targets, alpha=1.0)
            else:
                mixed_x, y_a, y_b, lam = mixup_data(inputs, targets, alpha=0.8)
            outputs = model(mixed_x)
            loss = mixed_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            raise ValueError(f"Unknown regime: {regime}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, (correct / total) * 100.0


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, List[float]]:
    """Evaluates validation loss, top-1 accuracy, and prediction maximum probabilities."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    max_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1)
            batch_max_probs, preds = probs.max(dim=1)
            max_probs.extend(batch_max_probs.cpu().numpy())

            running_loss += loss.item() * inputs.size(0)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / total
    val_acc = (correct / total) * 100.0
    return val_loss, val_acc, max_probs


# -----------------------------------------------------------------------------
# Diagnostic Plotting Dashboard
# -----------------------------------------------------------------------------
def plot_augmentation_dashboard(
    experiment_results: Dict[str, Dict[str, any]],
    test_dataset: datasets.CIFAR10,
    save_path: Path,
) -> None:
    """Generates a publication-grade 4-panel evaluation dashboard."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # 1. Qualitative Augmentation Gallery (MixUp and CutMix Visual Samples)
    ax_gallery = axes[0, 0]
    ax_gallery.axis("off")
    ax_gallery.set_title("(a) CutMix & MixUp Synthesized Samples", fontsize=13, fontweight="bold", pad=10)

    sub_spec = axes[0, 0].get_subplotspec()
    inner_grid = fig.add_subfigure(sub_spec)
    sub_axes = inner_grid.subplots(2, 2)

    # Pick two distinct samples
    raw_img1 = test_dataset.data[12]  # horse
    lbl1 = CLASSES[test_dataset.targets[12]]
    raw_img2 = test_dataset.data[29]  # frog
    lbl2 = CLASSES[test_dataset.targets[29]]

    t1 = torch.from_numpy(raw_img1).permute(2, 0, 1).float() / 255.0
    t2 = torch.from_numpy(raw_img2).permute(2, 0, 1).float() / 255.0
    pair_batch = torch.stack([t1, t2])
    pair_labels = torch.tensor([test_dataset.targets[12], test_dataset.targets[29]])

    # Sub 1: Original Image 1
    sub_axes[0, 0].imshow(raw_img1)
    sub_axes[0, 0].set_title(f"Original A: {lbl1}", fontsize=9, fontweight="semibold")
    sub_axes[0, 0].axis("off")

    # Sub 2: Original Image 2
    sub_axes[0, 1].imshow(raw_img2)
    sub_axes[0, 1].set_title(f"Original B: {lbl2}", fontsize=9, fontweight="semibold")
    sub_axes[0, 1].axis("off")

    # Sub 3: MixUp Blended
    mixed_up, _, _, lam_up = mixup_data(pair_batch, pair_labels, alpha=1.0)
    img_up = mixed_up[0].permute(1, 2, 0).numpy().clip(0, 1)
    sub_axes[1, 0].imshow(img_up)
    sub_axes[1, 0].set_title(f"MixUp ({lam_up*100:.0f}% {lbl1} + {(1-lam_up)*100:.0f}% {lbl2})", fontsize=8.5, fontweight="semibold")
    sub_axes[1, 0].axis("off")

    # Sub 4: CutMix Patched
    cut_mixed, _, _, lam_cut = cutmix_data(pair_batch, pair_labels, alpha=1.0)
    img_cut = cut_mixed[0].permute(1, 2, 0).numpy().clip(0, 1)
    sub_axes[1, 1].imshow(img_cut)
    sub_axes[1, 1].set_title(f"CutMix ({lam_cut*100:.0f}% {lbl1} + {(1-lam_cut)*100:.0f}% {lbl2})", fontsize=8.5, fontweight="semibold")
    sub_axes[1, 1].axis("off")

    # 2. Validation Loss Trajectories
    ax_loss = axes[0, 1]
    epochs = range(1, NUM_EPOCHS + 1)
    colors = {"baseline": "#e41a1c", "mixup": "#377eb8", "cutmix": "#ff7f00", "cutmix_mixup": "#4daf4a"}
    labels = {
        "baseline": "Baseline (Standard)",
        "mixup": "MixUp (alpha=0.8)",
        "cutmix": "CutMix (alpha=1.0)",
        "cutmix_mixup": "CutMix + MixUp (50/50)",
    }

    for key, data in experiment_results.items():
        ax_loss.plot(epochs, data["val_loss"], "o-", label=labels[key], color=colors[key], linewidth=2.0)
    ax_loss.set_title("(b) Validation Loss Progression", fontsize=13, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax_loss.set_xticks(list(epochs))
    ax_loss.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    # 3. Validation Top-1 Accuracy Trajectories
    ax_acc = axes[1, 0]
    for key, data in experiment_results.items():
        ax_acc.plot(epochs, data["val_acc"], "s--", label=labels[key], color=colors[key], linewidth=2.0)
    ax_acc.set_title("(c) Validation Top-1 Accuracy (%)", fontsize=13, fontweight="bold", pad=10)
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_xticks(list(epochs))
    ax_acc.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax_acc.grid(True, linestyle=":", alpha=0.6)

    # 4. Prediction Confidence Distribution (Overconfidence Mitigation)
    ax_conf = axes[1, 1]
    for key, data in experiment_results.items():
        probs = data["max_probs"]
        ax_conf.hist(probs, bins=25, alpha=0.45, label=labels[key], color=colors[key], density=True)
    ax_conf.set_title("(d) Model Confidence Distribution (Max Softmax Probability)", fontsize=13, fontweight="bold", pad=10)
    ax_conf.set_xlabel("Maximum Class Probability $P_{\max}$", fontsize=11)
    ax_conf.set_ylabel("Density", fontsize=11)
    ax_conf.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.0)
    ax_conf.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Artifact] Saved CutMix & MixUp dashboard to: {save_path}")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("Project 37: CutMix and MixUp Augmentation Strategies on CIFAR-10")
    print("=" * 75)
    print(f"Compute Device        : {device}")
    print(f"Backbone Model        : MiniResNet (4-Stage Residual CNN)")
    print(f"Dataset               : CIFAR-10 ({NUM_TRAIN_SAMPLES} train, {NUM_VAL_SAMPLES} val)")
    print(f"Augmentation Regimes  : Baseline, MixUp, CutMix, CutMix+MixUp")
    print(f"Hyperparameters       : Epochs={NUM_EPOCHS} | Batch Size={BATCH_SIZE} | LR=1e-3")
    print("-" * 75)

    # 1. Dataset
    train_loader, test_loader, full_test = get_cifar10_dataloaders(DATA_DIR)

    # 2. Frozen Initial Weights for Strict Experimental Fairness
    canonical_model = MiniResNet().to(device)
    base_state_dict = copy.deepcopy(canonical_model.state_dict())
    total_params = sum(p.numel() for p in canonical_model.parameters())
    print(f"Model Parameters      : {total_params:,} (Identical initialization for all runs)")
    print("-" * 75)

    # 3. Benchmark Augmentation Regimes
    regimes = ["baseline", "mixup", "cutmix", "cutmix_mixup"]
    experiment_results = {}
    best_overall_acc = 0.0
    best_overall_state = None
    best_regime_name = ""

    for idx, regime in enumerate(regimes, 1):
        print(f"\n>>> [{idx}/{len(regimes)}] Training Regime: {regime.upper()}...")
        model = MiniResNet().to(device)
        model.load_state_dict(copy.deepcopy(base_state_dict))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

        regime_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        start_t = time.perf_counter()

        for epoch in range(1, NUM_EPOCHS + 1):
            ep_start = time.perf_counter()
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, regime, device)
            v_loss, v_acc, max_probs = evaluate(model, test_loader, criterion, device)
            scheduler.step()

            regime_history["train_loss"].append(t_loss)
            regime_history["train_acc"].append(t_acc)
            regime_history["val_loss"].append(v_loss)
            regime_history["val_acc"].append(v_acc)

            ep_time = time.perf_counter() - ep_start
            print(
                f"    Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] "
                f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
                f"Val Acc: {v_acc:6.2f}% | Time: {ep_time:.2f}s"
            )

        total_duration = time.perf_counter() - start_t
        regime_history["total_time_s"] = total_duration
        regime_history["max_probs"] = max_probs
        experiment_results[regime] = regime_history

        final_acc = regime_history["val_acc"][-1]
        if final_acc > best_overall_acc:
            best_overall_acc = final_acc
            best_overall_state = copy.deepcopy(model.state_dict())
            best_regime_name = regime

    # 4. Save Checkpoint of Top Model
    checkpoint_path = CHECKPOINT_DIR / "best_cutmix_mixup_model.pt"
    torch.save(
        {
            "best_regime": best_regime_name,
            "best_val_acc": best_overall_acc,
            "model_state_dict": best_overall_state,
        },
        checkpoint_path,
    )
    print(f"\n[Checkpoint] Saved best model ({best_regime_name}) to: {checkpoint_path}")

    # 5. Diagnostic Plotting
    figure_path = PROJECT_DIR / "cutmix_mixup_augmentation_result.png"
    plot_augmentation_dashboard(experiment_results, full_test, figure_path)

    print("\n" + "=" * 75)
    print("CutMix & MixUp Experiment Completed Successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()

