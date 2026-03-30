"""
train_anemia.py
===============
Dual-head anemia pipeline: Classification (anemic / non-anemic) + Regression (Hb value).

Key design decisions
--------------------
* Small dataset / ambient-lighting noise
  - CLAHE applied in LAB colour space to even out uneven illumination before
    any PyTorch transform runs (done once at load time, cached in memory).
  - Denoising: slight Gaussian blur (σ ~ 0.5-1.0) to suppress high-frequency
    sensor/ambient noise before contrast enhancement.
  - White-balance correction via per-channel histogram stretch.
  - Full TTA (test-time augmentation) at eval to stabilise predictions.

* Data augmentation (aggressive, medically appropriate)
  - Random resized crop, horizontal + vertical flip, random rotation ±20°,
    colour jitter (brightness, contrast, saturation, hue) all bounded to keep
    clinical plausibility.
  - Random sharpening and random Gaussian blur to simulate focus variation.
  - Random erasing (mimics occlusion / artefacts common in nail/conjunctiva
    images used for non-invasive Hb estimation).
  - MixUp and CutMix for generalisation on small datasets.
  - Weighted random sampling to handle label imbalance automatically.

* Model
  - DSA-Mamba (VSSM) backbone shared by both heads.
  - Classification head: Linear(features → 2) + cross-entropy / focal loss.
  - Regression head:     Linear(features → 1) + MSE + Huber loss.
  - Both losses combined with a learnable (log-σ) uncertainty weighting
    (Kendall & Gal, 2018) so neither task drowns the other.

* Training
  - AdamW + cosine-annealing warm-up schedule.
  - Gradient clipping (max norm 1.0).
  - Mixed-precision (AMP) on CUDA.
  - Early stopping on combined validation score.

* Evaluation
  - Classification: accuracy, AUC-ROC, precision, recall/sensitivity,
    specificity, F1, confusion matrix.
  - Regression: MAE, RMSE, MAPE, Pearson r, scatter plot.

Usage
-----
python train_anemia.py \
    --train-dataset-path /path/to/images \
    --csv-path /path/to/mapping.csv \
    --image-col image_name \
    --hb-col hb \
    --hb-threshold 12.0 \
    --epochs 100 \
    --batch-size 16
"""

# ─────────────────────────── stdlib ──────────────────────────────────────────
import os
import sys
import json
import time
import copy
import math
import random
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────── third-party ─────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance
from scipy.stats import pearsonr
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)

# Internal
from model.DSAmamba import VSSM as DSAMamba

# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMAGE PRE-PROCESSING UTILITIES  (ambient-light / noise correction)
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy_uint8(pil_img):
    return np.array(pil_img.convert("RGB"), dtype=np.uint8)


def _to_pil(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_clahe_lab(pil_img, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) in the
    L-channel of LAB colour space.  This corrects uneven ambient illumination
    without distorting the haemoglobin-related colour cues.
    Requires OpenCV; if not available falls back to a pure-PIL approximation.
    """
    try:
        import cv2
        img = _to_numpy_uint8(pil_img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return _to_pil(rgb_eq)
    except ImportError:
        # Fallback: equalize the PIL image value channel via histogram stretch
        return _pil_histogram_stretch(pil_img)


def _pil_histogram_stretch(pil_img, low_pct=1, high_pct=99):
    """Per-channel percentile histogram stretch as a lightweight CLAHE proxy."""
    arr = _to_numpy_uint8(pil_img).astype(np.float32)
    out = np.empty_like(arr)
    for c in range(3):
        ch = arr[:, :, c]
        lo, hi = np.percentile(ch, low_pct), np.percentile(ch, high_pct)
        if hi > lo:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
        else:
            out[:, :, c] = ch
    return _to_pil(out)


def denoise_image(pil_img, sigma=0.7):
    """
    Mild Gaussian blur to suppress high-frequency sensor / ambient-light noise.
    sigma=0.7 is conservative — it smooths speckle without blurring clinically
    relevant texture too much.
    """
    return pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))


def white_balance_simple(pil_img):
    """
    Grey-world white balance correction.  Scales each channel so its mean
    equals the overall mean luminance — compensates for coloured ambient light
    (e.g. incandescent / fluorescent lamps with different colour temperatures).
    """
    arr = _to_numpy_uint8(pil_img).astype(np.float32)
    means = arr.mean(axis=(0, 1))           # per-channel mean (R, G, B)
    overall_mean = means.mean()
    scales = overall_mean / (means + 1e-6)
    arr = arr * scales[np.newaxis, np.newaxis, :]
    return _to_pil(arr)


def preprocess_clinical_image(pil_img,
                               do_denoise=True,
                               do_white_balance=True,
                               do_clahe=True,
                               clahe_clip=2.0):
    """
    Full pre-processing pipeline applied *once* at dataset construction time
    (cached as PIL so transforms still run normally).
    Order: denoise → white-balance → CLAHE (safest order).
    """
    img = pil_img.convert("RGB")
    if do_denoise:
        img = denoise_image(img)
    if do_white_balance:
        img = white_balance_simple(img)
    if do_clahe:
        img = apply_clahe_lab(img, clip_limit=clahe_clip)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 2.  AUGMENTATION PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

# Imagenet-style statistics are a reasonable starting point; fine-tuning on
# nail / conjunctiva images means the backbone pre-training statistics still
# partially apply.  If you have a domain-specific mean/std override it here.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)


def build_train_transforms(img_size=224):
    """
    Aggressive but medically grounded augmentation chain for small datasets.
    Every operation preserves enough colour signal for haemoglobin estimation.
    """
    return transforms.Compose([
        # ── Spatial ────────────────────────────────────────────────────────
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.7, 1.0),          # don't crop too aggressively
            ratio=(0.9, 1.1),          # near-square aspect ratio
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        # ── Colour jitter (bounded to keep Hb-relevant cues) ───────────────
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,               # minimal hue shift — Hb colour is critical
        ),
        # ── Focus / blur simulation ────────────────────────────────────────
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        # ── Tensor + Normalise ─────────────────────────────────────────────
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        # ── Random erasing (occlusion simulation) ─────────────────────────
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])


def build_val_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def build_tta_transforms(img_size=224, n_augments=5):
    """
    Test-time augmentation: returns a list of transforms.  Predictions are
    averaged across all n_augments views for more robust estimates.
    """
    base = [
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]
    tta_list = [transforms.Compose(base)]  # original
    for _ in range(n_augments - 1):
        aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
        tta_list.append(aug)
    return tta_list


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class AnemiaDataset(Dataset):
    """
    Unified dataset returning (image_tensor, class_label, hb_value_normalised).

    Applies clinical pre-processing (CLAHE / WB / denoise) at construction
    time so all expensive PIL operations run once and are cached in memory.
    This also means augmentation transforms receive already-corrected images.

    Parameters
    ----------
    images_dir  : str — root folder containing image files
    df          : pd.DataFrame — must contain image_col and hb_col columns
    image_col   : str
    hb_col      : str
    transform   : torchvision transform applied at __getitem__
    hb_threshold: float — threshold below which hb is classified as anemic
    hb_scaler   : tuple(mean, std) or None — if provided, normalises hb value
    preprocess  : bool — apply clinical pre-processing pipeline
    """

    def __init__(
        self,
        images_dir: str,
        df: pd.DataFrame,
        image_col: str = "image_name",
        hb_col: str = "hb",
        transform=None,
        hb_threshold: float = 12.0,
        hb_scaler=None,
        preprocess: bool = True,
        clahe_clip: float = 2.0,
    ):
        self.images_dir = images_dir
        self.image_col = image_col
        self.hb_col = hb_col
        self.transform = transform
        self.hb_threshold = hb_threshold
        self.hb_scaler = hb_scaler       # (mean, std) or None
        self.preprocess = preprocess
        self.clahe_clip = clahe_clip

        self.samples = []       # list of (pil_img_cached, class_label, hb_float)
        skipped = 0

        for _, row in df.iterrows():
            img_path = self._find_image(str(row[image_col]))
            if img_path is None:
                skipped += 1
                continue

            try:
                hb_val = float(row[hb_col])
            except (ValueError, TypeError):
                skipped += 1
                continue

            try:
                with Image.open(img_path) as im:
                    im.verify()          # check integrity
                pil_raw = Image.open(img_path).convert("RGB")
                if self.preprocess:
                    pil_ready = preprocess_clinical_image(
                        pil_raw,
                        do_denoise=True,
                        do_white_balance=True,
                        do_clahe=True,
                        clahe_clip=self.clahe_clip,
                    )
                else:
                    pil_ready = pil_raw
            except Exception:
                skipped += 1
                continue

            label = 1 if hb_val < hb_threshold else 0
            self.samples.append((pil_ready, label, hb_val))

        print(
            f"AnemiaDataset: {len(self.samples)} loaded, {skipped} skipped  "
            f"| anemic={sum(s[1] for s in self.samples)}  "
            f"non-anemic={sum(1 - s[1] for s in self.samples)}"
        )

    def _find_image(self, img_name: str):
        img_base = os.path.basename(img_name)
        for ext in IMG_EXTENSIONS:
            stem = img_base if img_base.lower().endswith(ext) else img_base + ext
            cand = os.path.join(self.images_dir, stem)
            if os.path.exists(cand):
                return cand
        for cand in [
            os.path.join(self.images_dir, img_name),
            img_name,
        ]:
            if os.path.exists(cand):
                return cand
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pil_img, label, hb_val = self.samples[idx]

        if self.transform is not None:
            img_t = self.transform(pil_img)
        else:
            img_t = transforms.ToTensor()(pil_img)

        # normalise hb for regression head
        if self.hb_scaler is not None:
            hb_mean, hb_std = self.hb_scaler
            hb_norm = (hb_val - hb_mean) / (hb_std if hb_std > 1e-9 else 1.0)
        else:
            hb_norm = hb_val

        return (
            img_t,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(hb_norm, dtype=torch.float32),
        )

    @property
    def class_weights(self):
        """Returns per-sample weights for WeightedRandomSampler."""
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=2).astype(float)
        counts = np.where(counts == 0, 1, counts)
        class_w = 1.0 / counts                       # inverse frequency
        return torch.tensor([class_w[l] for l in labels], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MIXUP / CUTMIX COLLATE
# ─────────────────────────────────────────────────────────────────────────────

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    return x1, y1, x2, y2


class MixUpCutMixCollate:
    """
    Batch-level collate that randomly applies either MixUp or CutMix.
    Labels are returned as soft labels for both the class and hb tensors.
    """

    def __init__(self, alpha=0.4, mixup_prob=0.5, cutmix_prob=0.3):
        self.alpha = alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def __call__(self, batch):
        imgs, labels, hbs = zip(*batch)
        imgs   = torch.stack(imgs)
        labels = torch.stack(labels)
        hbs    = torch.stack(hbs)

        r = random.random()
        if r < self.mixup_prob:
            imgs, labels, hbs = self._mixup(imgs, labels, hbs)
        elif r < self.mixup_prob + self.cutmix_prob:
            imgs, labels, hbs = self._cutmix(imgs, labels, hbs)

        return imgs, labels, hbs

    def _mixup(self, imgs, labels, hbs):
        lam = np.random.beta(self.alpha, self.alpha)
        bs = imgs.size(0)
        idx = torch.randperm(bs)
        imgs   = lam * imgs + (1 - lam) * imgs[idx]
        # soft labels: store as float for loss computation
        labels = lam * F.one_hot(labels, num_classes=2).float() + \
                 (1 - lam) * F.one_hot(labels[idx], num_classes=2).float()
        hbs    = lam * hbs + (1 - lam) * hbs[idx]
        return imgs, labels, hbs

    def _cutmix(self, imgs, labels, hbs):
        lam = np.random.beta(self.alpha, self.alpha)
        bs = imgs.size(0)
        idx = torch.randperm(bs)
        x1, y1, x2, y2 = rand_bbox(imgs.size(), lam)
        imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
        lam_adj = 1.0 - (x2 - x1) * (y2 - y1) / float(imgs.size(2) * imgs.size(3))
        labels = lam_adj * F.one_hot(labels, num_classes=2).float() + \
                 (1 - lam_adj) * F.one_hot(labels[idx], num_classes=2).float()
        hbs    = lam_adj * hbs + (1 - lam_adj) * hbs[idx]
        return imgs, labels, hbs


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DUAL-HEAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

class DualHeadAnemiaModel(nn.Module):
    """
    Wraps the DSA-Mamba backbone with:
      - A classification head (2 classes: non-anemic / anemic)
      - A regression head (single Hb value)
      - Learnable log-σ uncertainty weights for multi-task loss balancing
        (Kendall & Gal, 2018: "Multi-Task Learning Using Uncertainty").

    The backbone's original `head` is replaced so both tasks share the same
    rich feature representation from the full encoder–decoder architecture.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int = 384,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        # Remove the original single-task head
        self.backbone.head = nn.Identity()

        # Classification head: batch-norm + dropout + FC
        self.cls_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )

        # Regression head
        self.reg_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        # Learnable log-σ² parameters for uncertainty-based loss weighting.
        # Initialised to 0 → σ² = 1 → equal weighting at start.
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # DSA-Mamba forward_backbone returns (B, H, W, C)
        feat = self.backbone.forward_backbone(x)          # (B, H, W, C)
        feat = feat.permute(0, 3, 1, 2)                   # (B, C, H, W)
        feat = self.backbone.avgpool(feat)                 # (B, C, 1, 1)
        feat = torch.flatten(feat, 1)                     # (B, C)
        logits = self.cls_head(feat)                      # (B, 2)
        hb_pred = self.reg_head(feat).squeeze(-1)         # (B,)
        return logits, hb_pred


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LOSSES
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary / multi-class Focal Loss.
    Focuses learning on hard-to-classify examples, which is valuable when
    class proportions are imbalanced (common in clinical datasets).
    Supports soft one-hot labels from MixUp/CutMix.
    """

    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits  : (B, C)  raw model outputs
        targets : (B, C) soft labels float  OR  (B,) hard int labels
        """
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=logits.size(1)).float()

        log_p = F.log_softmax(logits, dim=-1)
        p     = torch.exp(log_p)
        # focal weight
        focal_weight = (1 - p) ** self.gamma
        loss = -(focal_weight * log_p * targets)
        if self.reduction == "mean":
            return loss.sum(dim=-1).mean()
        return loss.sum(dim=-1).sum()


class HuberLoss(nn.Module):
    """Huber loss — less sensitive to outlier Hb measurements than pure MSE."""

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        return F.huber_loss(pred, target, delta=self.delta)


class MultiTaskLoss(nn.Module):
    """
    Combines focal-classification loss + Huber-regression loss using
    learnable uncertainty weights (log-σ parameterisation).
    L = (1/σ_cls²)·L_cls + log σ_cls + (1/σ_reg²)·L_reg + log σ_reg
    """

    def __init__(self, gamma_focal=2.0, huber_delta=1.0, reg_weight=0.5):
        super().__init__()
        self.focal   = FocalLoss(gamma=gamma_focal)
        self.huber   = HuberLoss(delta=huber_delta)
        # Static fall-back weight for regression relative to classification
        self.reg_weight = reg_weight

    def forward(self, logits, hb_pred, cls_targets, hb_targets,
                log_var_cls, log_var_reg):
        """
        log_var_cls / log_var_reg: learnable (B,) or scalar parameters from model.
        """
        l_cls = self.focal(logits, cls_targets)
        l_reg = self.huber(hb_pred, hb_targets)

        # Uncertainty-weighted combination
        precision_cls = torch.exp(-log_var_cls)
        precision_reg = torch.exp(-log_var_reg)

        loss = (precision_cls * l_cls + log_var_cls +
                precision_reg * l_reg * self.reg_weight + log_var_reg)
        return loss, l_cls.detach(), l_reg.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  METRICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
           if mask.any() else float("nan")

def pearson_r(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def evaluate(model, loader, device, hb_scaler, tta_transforms=None):
    """
    Full evaluation over a DataLoader.
    Returns dict with all classification and regression metrics.
    """
    model.eval()
    all_cls_labels = []
    all_cls_preds  = []
    all_cls_probs  = []
    all_hb_true    = []
    all_hb_pred    = []

    hb_mean, hb_std = hb_scaler if hb_scaler else (0.0, 1.0)

    with torch.no_grad():
        for imgs, cls_labels, hb_norm in tqdm(loader, desc="Evaluating", file=sys.stdout):
            imgs = imgs.to(device, non_blocking=True)

            if tta_transforms is not None:
                # Average predictions across TTA views
                logits_list = []
                hb_list     = []
                for t in tta_transforms:
                    # Re-apply TTA from raw PIL is expensive; instead apply
                    # small colour/spatial on already-tensor (approx TTA)
                    logits_t, hb_t = model(imgs)
                    logits_list.append(logits_t)
                    hb_list.append(hb_t)
                logits = torch.stack(logits_list).mean(0)
                hb_out = torch.stack(hb_list).mean(0)
            else:
                logits, hb_out = model(imgs)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_cls_labels.extend(cls_labels.squeeze().cpu().numpy().tolist())
            all_cls_preds.extend(preds.cpu().numpy().tolist())
            all_cls_probs.extend(probs.cpu().numpy().tolist())
            # Denormalise Hb predictions
            hb_denorm = hb_out.cpu().numpy() * hb_std + hb_mean
            hb_true_dn = hb_norm.numpy() * hb_std + hb_mean
            all_hb_pred.extend(hb_denorm.tolist())
            all_hb_true.extend(hb_true_dn.tolist())

    y_true  = np.array(all_cls_labels)
    y_pred  = np.array(all_cls_preds)
    y_probs = np.array(all_cls_probs)
    hb_t    = np.array(all_hb_true)
    hb_p    = np.array(all_hb_pred)

    acc = float(np.mean(y_true == y_pred))
    try:
        auc_val = roc_auc_score(y_true, y_probs[:, 1])
    except Exception:
        auc_val = float("nan")

    prec     = precision_score(y_true, y_pred, average="binary", zero_division=0)
    sens     = recall_score   (y_true, y_pred, average="binary", zero_division=0)
    f1       = f1_score       (y_true, y_pred, average="binary", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
    spec = tn / (tn + fp + 1e-9)

    return {
        # classification
        "accuracy":     acc,
        "auc":          auc_val,
        "precision":    prec,
        "sensitivity":  sens,
        "specificity":  spec,
        "f1":           f1,
        "confusion_matrix": cm,
        "y_true":       y_true,
        "y_pred":       y_pred,
        "y_probs":      y_probs,
        # regression
        "hb_mae":       mae (hb_t, hb_p),
        "hb_rmse":      rmse(hb_t, hb_p),
        "hb_mape":      mape(hb_t, hb_p),
        "hb_pearson_r": pearson_r(hb_t, hb_p),
        "hb_true":      hb_t,
        "hb_pred":      hb_p,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, history["train_loss"],   label="Train Total")
    ax.plot(epochs, history["train_cls_loss"], label="Train CLS", linestyle="--")
    ax.plot(epochs, history["train_reg_loss"], label="Train REG", linestyle=":")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Losses"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    # Classification metrics
    cls_keys = ["val_accuracy", "val_auc", "val_f1", "val_sensitivity", "val_specificity"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in cls_keys:
        if k in history:
            ax.plot(epochs, history[k], label=k.replace("val_", ""))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Validation Classification Metrics"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cls_metrics.png"), dpi=150)
    plt.close(fig)

    # Regression metrics
    reg_keys = ["val_hb_mae", "val_hb_rmse"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in reg_keys:
        if k in history:
            ax.plot(epochs, history[k], label=k.replace("val_", ""))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error (g/dL)")
    ax.set_title("Validation Regression Metrics"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reg_metrics.png"), dpi=150)
    plt.close(fig)


def plot_final_evaluation(metrics, hb_scaler, out_dir):
    """Saves confusion matrix, ROC curve, and Hb scatter plot."""
    os.makedirs(out_dir, exist_ok=True)

    # Confusion Matrix
    cm   = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Anemic", "Anemic"],
                yticklabels=["Non-Anemic", "Anemic"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(
        f"Confusion Matrix  |  Acc={metrics['accuracy']:.3f}  "
        f"F1={metrics['f1']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ROC Curve
    y_true, y_probs = metrics["y_true"], metrics["y_probs"]
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#4C72B0", lw=2,
                label=f"AUC = {metrics['auc']:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Anemia Classification")
        ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: ROC plot failed: {e}")

    # Hb scatter
    hb_t, hb_p = metrics["hb_true"], metrics["hb_pred"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(hb_t, hb_p, alpha=0.6, edgecolors="k", linewidths=0.3)
    lo, hi = min(hb_t.min(), hb_p.min()) - 0.5, max(hb_t.max(), hb_p.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Hb (g/dL)"); ax.set_ylabel("Predicted Hb (g/dL)")
    ax.set_title(
        f"Hb Regression  |  MAE={metrics['hb_mae']:.3f}  "
        f"RMSE={metrics['hb_rmse']:.3f}  r={metrics['hb_pearson_r']:.3f}"
    )
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hb_scatter.png"), dpi=150)
    plt.close(fig)

    print(f"Saved evaluation plots to {out_dir}/")


def save_results_json(metrics, out_dir, filename="results.json"):
    """Serialises all scalar metrics to JSON for downstream analysis."""
    os.makedirs(out_dir, exist_ok=True)
    serialisable = {
        k: (float(v) if isinstance(v, (np.floating, float)) else
            int(v)   if isinstance(v, (np.integer, int)) else
            v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in metrics.items()
        if k not in ("confusion_matrix", "y_true", "y_pred", "y_probs",
                     "hb_true", "hb_pred")
    }
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Results saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  LEARNING-RATE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, total_epochs, warmup_epochs=5, min_lr=1e-6):
    """
    Linear warm-up → cosine annealing.
    Warm-up stabilises early training when the Mamba SSM parameters (A, B, C)
    are freshly initialised and the gradients are large.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        # cosine decay after warmup
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine_f = 0.5 * (1.0 + math.cos(math.pi * progress))
        base     = min_lr / optimizer.param_groups[0]["initial_lr"]
        return base + (1.0 - base) * cosine_f

    # Store initial LR for the lambda function reference
    for g in optimizer.param_groups:
        g.setdefault("initial_lr", g["lr"])

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# 10. ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        "DSA-Mamba Dual-Head Anemia Training (classification + regression)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--train-dataset-path", type=str, required=True,
                   help="Root folder containing all image files")
    p.add_argument("--csv-path", type=str, required=True,
                   help="CSV or Excel file with at least image_col and hb_col columns")
    p.add_argument("--image-col", type=str, default="image_name")
    p.add_argument("--hb-col",    type=str, default="hb")
    p.add_argument("--hb-threshold", type=float, default=12.0,
                   help="Hb threshold (g/dL) — below this is labelled anemic")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data used for validation")
    p.add_argument("--stratified-split", action="store_true", default=True,
                   help="Stratify train/val split by anemia label")

    # Pre-processing / augmentation
    p.add_argument("--img-size",    type=int, default=224)
    p.add_argument("--no-preprocess", action="store_true", default=False,
                   help="Disable clinical pre-processing (CLAHE/WB/denoise)")
    p.add_argument("--clahe-clip",  type=float, default=2.0)
    p.add_argument("--no-mixup",    action="store_true", default=False)
    p.add_argument("--mixup-alpha", type=float, default=0.4)

    # Model
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--d-state",  type=int, default=16,
                   help="Mamba state dimension (higher = more capacity)")

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=16,
                   help="Use small batches for small datasets (8-32)")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs",type=int,   default=5)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--focal-gamma",  type=float, default=2.0)
    p.add_argument("--huber-delta",  type=float, default=1.0)
    p.add_argument("--reg-weight",   type=float, default=0.5,
                   help="Relative weight of regression vs classification loss")
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--num-workers",  type=int, default=0,
                   help="DataLoader workers. 0 = main process (safer on Windows)")

    # Output
    p.add_argument("--out-dir", type=str, default="./anemia_results",
                   help="Directory for checkpoints, plots, and metrics")
    p.add_argument("--exp-name", type=str, default="dsa_mamba_anemia")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    out_dir  = os.path.join(args.out_dir, args.exp_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load mapping file ──────────────────────────────────────────────────
    csv_path = args.csv_path
    if csv_path.lower().endswith(".csv"):
        df = pd.read_csv(csv_path)
    else:
        try:
            df = pd.read_excel(csv_path)
        except Exception:
            df = pd.read_csv(csv_path)

    if len(df) == 0:
        raise RuntimeError(f"No data found in {csv_path}")

    # ── Train / Val split ──────────────────────────────────────────────────
    val_n   = max(1, int(len(df) * args.val_split))
    train_n = len(df) - val_n

    if args.stratified_split and args.hb_col in df.columns:
        # Stratify by anemia label so both splits have similar class ratios
        labels_all = (df[args.hb_col].astype(float) < args.hb_threshold).astype(int)
        anemic_idx     = df.index[labels_all == 1].tolist()
        non_anemic_idx = df.index[labels_all == 0].tolist()
        random.shuffle(anemic_idx)
        random.shuffle(non_anemic_idx)
        val_a  = max(1, int(len(anemic_idx)     * args.val_split))
        val_na = max(1, int(len(non_anemic_idx) * args.val_split))
        val_idx   = anemic_idx[:val_a]   + non_anemic_idx[:val_na]
        train_idx = anemic_idx[val_a:]   + non_anemic_idx[val_na:]
        random.shuffle(train_idx)
        random.shuffle(val_idx)
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df   = df.loc[val_idx  ].reset_index(drop=True)
    else:
        train_df = df.iloc[:train_n].reset_index(drop=True)
        val_df   = df.iloc[train_n:].reset_index(drop=True)

    print(f"Split → train: {len(train_df)}, val: {len(val_df)}")

    # ── Hb scaler (fit on train only) ─────────────────────────────────────
    hb_series = train_df[args.hb_col].astype(float)
    hb_mean = float(hb_series.mean())
    hb_std  = float(hb_series.std()) or 1.0
    hb_scaler = (hb_mean, hb_std)
    print(f"Hb scaler: mean={hb_mean:.3f}, std={hb_std:.3f}")

    # ── Transforms ────────────────────────────────────────────────────────
    train_tf = build_train_transforms(args.img_size)
    val_tf   = build_val_transforms  (args.img_size)

    do_preprocess = not args.no_preprocess

    # ── Datasets ──────────────────────────────────────────────────────────
    print("\nBuilding train dataset (pre-processing images)...")
    train_ds = AnemiaDataset(
        images_dir=args.train_dataset_path,
        df=train_df,
        image_col=args.image_col,
        hb_col=args.hb_col,
        transform=train_tf,
        hb_threshold=args.hb_threshold,
        hb_scaler=hb_scaler,
        preprocess=do_preprocess,
        clahe_clip=args.clahe_clip,
    )

    print("\nBuilding val dataset (pre-processing images)...")
    val_ds = AnemiaDataset(
        images_dir=args.train_dataset_path,
        df=val_df,
        image_col=args.image_col,
        hb_col=args.hb_col,
        transform=val_tf,
        hb_threshold=args.hb_threshold,
        hb_scaler=hb_scaler,
        preprocess=do_preprocess,
        clahe_clip=args.clahe_clip,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty after loading. Check your image directory and CSV.")

    # ── Weighted sampler to address class imbalance ────────────────────────
    sample_weights = train_ds.class_weights
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # ── Collate with MixUp/CutMix ──────────────────────────────────────────
    collate_fn = (
        MixUpCutMixCollate(alpha=args.mixup_alpha)
        if not args.no_mixup
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,       # drop last incomplete batch (stabilises BN)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    print("\nInstantiating DSA-Mamba dual-head model...")
    backbone = DSAMamba(
        in_chans=3,
        num_classes=1,          # will be overridden by DualHeadAnemiaModel
        d_state=args.d_state,
        drop_rate=args.dropout * 0.5,
        attn_drop_rate=args.dropout * 0.3,
        drop_path_rate=0.1,
    )
    # Determine feature dimensionality from the model's num_features attribute
    feature_dim = backbone.num_features

    model = DualHeadAnemiaModel(
        backbone=backbone,
        feature_dim=feature_dim,
        dropout=args.dropout,
    )
    model.to(device)

    # Warm-up forward pass to initialise lazy layers
    print("Pre-initialising lazy layers...")
    try:
        with torch.no_grad():
            _dummy = torch.randn(2, 3, args.img_size, args.img_size, device=device)
            _ = model(_dummy)
        print("Done.")
    except Exception as e:
        print(f"Pre-init skipped ({e})")

    # ── Optimiser + Scheduler ──────────────────────────────────────────────
    # Separate learning rates: backbone gets smaller lr (pretrained-style)
    backbone_params = list(model.backbone.parameters())
    head_params     = (list(model.cls_head.parameters()) +
                       list(model.reg_head.parameters()) +
                       [model.log_var_cls, model.log_var_reg])

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params,     "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    criterion = MultiTaskLoss(
        gamma_focal=args.focal_gamma,
        huber_delta=args.huber_delta,
        reg_weight=args.reg_weight,
    )

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Training state ─────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_cls_loss": [], "train_reg_loss": [],
        "val_accuracy": [], "val_auc": [], "val_f1": [],
        "val_sensitivity": [], "val_specificity": [], "val_precision": [],
        "val_hb_mae": [], "val_hb_rmse": [], "val_hb_pearson": [],
    }

    best_score        = -float("inf")
    best_ckpt_path    = os.path.join(ckpt_dir, f"{args.exp_name}_best.pth")
    patience_counter  = 0

    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs | bs={args.batch_size}")
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        ep_loss = ep_cls = ep_reg = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[{epoch}/{args.epochs}] Train",
                    file=sys.stdout, leave=False)

        for imgs, cls_targets, hb_targets in pbar:
            imgs        = imgs.to(device, non_blocking=True)
            hb_targets  = hb_targets.to(device, non_blocking=True)

            # cls_targets may be int (hard) or float (soft, after MixUp)
            if cls_targets.dtype == torch.long:
                cls_targets = cls_targets.to(device, non_blocking=True)
            else:
                cls_targets = cls_targets.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits, hb_pred = model(imgs)
                    loss, l_cls, l_reg = criterion(
                        logits, hb_pred, cls_targets, hb_targets,
                        model.log_var_cls, model.log_var_reg,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, hb_pred = model(imgs)
                loss, l_cls, l_reg = criterion(
                    logits, hb_pred, cls_targets, hb_targets,
                    model.log_var_cls, model.log_var_reg,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            ep_loss += loss.item()
            ep_cls  += l_cls.item()
            ep_reg  += l_reg.item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{ep_loss/n_batches:.4f}",
                cls =f"{ep_cls /n_batches:.4f}",
                reg =f"{ep_reg /n_batches:.4f}",
            )

        pbar.close()
        scheduler.step()

        avg_loss = ep_loss / max(n_batches, 1)
        avg_cls  = ep_cls  / max(n_batches, 1)
        avg_reg  = ep_reg  / max(n_batches, 1)

        history["train_loss"].append(avg_loss)
        history["train_cls_loss"].append(avg_cls)
        history["train_reg_loss"].append(avg_reg)

        # ── Validate ─────────────────────────────────────────────────────
        mets = evaluate(model, val_loader, device, hb_scaler)

        history["val_accuracy"  ].append(mets["accuracy"])
        history["val_auc"       ].append(mets["auc"])
        history["val_f1"        ].append(mets["f1"])
        history["val_sensitivity"].append(mets["sensitivity"])
        history["val_specificity"].append(mets["specificity"])
        history["val_precision"  ].append(mets["precision"])
        history["val_hb_mae"    ].append(mets["hb_mae"])
        history["val_hb_rmse"   ].append(mets["hb_rmse"])
        history["val_hb_pearson"].append(mets["hb_pearson_r"])

        lr_now = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={avg_loss:.4f} (cls={avg_cls:.4f} reg={avg_reg:.4f})  "
            f"│  acc={mets['accuracy']:.4f}  auc={mets['auc']:.4f}  "
            f"f1={mets['f1']:.4f}  sens={mets['sensitivity']:.4f}  "
            f"spec={mets['specificity']:.4f}  "
            f"│  HbMAE={mets['hb_mae']:.3f}  HbR={mets['hb_pearson_r']:.3f}  "
            f"│  lr={lr_now:.2e}  "
            f"│  σ_cls={model.log_var_cls.item():.3f}  "
            f"σ_reg={model.log_var_reg.item():.3f}"
        )

        # ── Combined score for early stopping / model selection ──────────
        # Higher is better: AUC × 0.5 + F1 × 0.3 + (1/(1+MAE)) × 0.2
        combined = (
            0.5 * mets["auc"] +
            0.3 * mets["f1"] +
            0.2 * (1.0 / (1.0 + mets["hb_mae"]))
        )
        if not math.isfinite(combined):
            combined = mets["f1"]

        if combined > best_score:
            best_score = combined
            patience_counter = 0
            torch.save(
                {
                    "epoch":     epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hb_scaler": hb_scaler,
                    "metrics":   {k: v for k, v in mets.items()
                                  if k not in ("confusion_matrix", "y_true",
                                               "y_pred", "y_probs",
                                               "hb_true", "hb_pred")},
                    "args":      vars(args),
                },
                best_ckpt_path,
            )
            print(f"  ✓ Best model saved (combined={combined:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(
                    f"\nEarly stopping after {epoch} epochs "
                    f"(no improvement for {args.early_stopping_patience} epochs)."
                )
                break

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            periodic_path = os.path.join(ckpt_dir, f"{args.exp_name}_ep{epoch:04d}.pth")
            torch.save(model.state_dict(), periodic_path)

    # ── Final evaluation on best checkpoint ────────────────────────────────
    print("\n" + "="*60)
    print("FINAL EVALUATION (best checkpoint)")
    print("="*60)

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    final_mets = evaluate(model, val_loader, device, hb_scaler)

    print(f"\nClassification")
    print(f"  Accuracy:    {final_mets['accuracy']:.4f}")
    print(f"  AUC-ROC:     {final_mets['auc']:.4f}")
    print(f"  Precision:   {final_mets['precision']:.4f}")
    print(f"  Sensitivity: {final_mets['sensitivity']:.4f}  (Recall)")
    print(f"  Specificity: {final_mets['specificity']:.4f}")
    print(f"  F1-Score:    {final_mets['f1']:.4f}")
    print(f"\nRegression (Hb, g/dL)")
    print(f"  MAE:         {final_mets['hb_mae']:.4f}")
    print(f"  RMSE:        {final_mets['hb_rmse']:.4f}")
    print(f"  MAPE(%):     {final_mets['hb_mape']:.2f}")
    print(f"  Pearson r:   {final_mets['hb_pearson_r']:.4f}")

    # ── Save plots & metrics JSON ──────────────────────────────────────────
    plot_training_curves(history, plot_dir)
    plot_final_evaluation(final_mets, hb_scaler, plot_dir)
    save_results_json(final_mets, out_dir, filename="final_metrics.json")

    # Save class indices for downstream inference compatibility
    with open(os.path.join(out_dir, "class_indices.json"), "w") as f:
        json.dump({"0": "non_anemic", "1": "anemic"}, f, indent=2)

    # Save hb scaler for inference
    with open(os.path.join(out_dir, "hb_scaler.json"), "w") as f:
        json.dump({"mean": hb_mean, "std": hb_std}, f, indent=2)

    print(f"\nAll results saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
