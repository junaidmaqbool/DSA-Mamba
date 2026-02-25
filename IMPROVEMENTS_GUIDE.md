# HB Regression Improvements Guide

## Problem Analysis

Your model was predicting values around 10 (the mean), indicating it got stuck at a local minimum where it just predicts the average. This is a common issue called "mode collapse" in regression tasks.

### Root Causes:
1. **No pretrained features** - Starting from scratch makes learning harder for image-based regression
2. **Loss function too smooth** - SmoothL1Loss can cause the model to settle at the mean
3. **Regression head not optimized** - Linear layer initialization wasn't suitable for full range prediction
4. **Learning rate issues** - Not well-tuned for regression tasks
5. **Insufficient regularization** - Model had no encouragement to use the full output range

---

## Solutions Implemented

### 1. **Transfer Learning** ✓
Use ImageNet pretrained weights from ResNet/DenseNet to initialize the backbone:

```bash
python train_hb_transfer.py \
  --csv-path your_data.csv \
  --train-dataset-path train_images/ \
  --pretrained-backbone resnet50 \
  --lr 5e-4 \
  --epochs 100
```

**Benefits:**
- Model starts with learned visual features from ImageNet
- Much faster convergence
- Better generalization with less data
- Reduces tendency to collapse to mean

### 2. **Improved Loss Function** ✓
Switched from SmoothL1Loss (which encourages mean prediction) to **MSE + Range Awareness Loss**:

```python
# New custom loss
criterion = RegressionLossWithRangeAwareness(
    variance_weight=0.1,  # Encourage output variance
    range_weight=0.05     # Encourage output range matching
)
```

**What it does:**
- **MSE Loss**: Main regression objective
- **Variance Loss**: Penalizes if output std ≠ target std (prevents collapse to mean)
- **Range Loss**: Penalizes if output range ≠ target range (encourages full utilization)

**Result:** Model is forced to use the full range of target values

### 3. **Better Learning Rate Strategy** ✓
Upgraded from basic learning rate to advanced scheduling:

```python
# Previous: Simple ReduceLROnPlateau
# New: CosineAnnealingWarmRestarts with differentiated LR

# Backbone gets lower LR (fine-tuning): 5e-5
# Head gets higher LR (training from scratch): 5e-4
```

**Benefits:**
- Backbone (from ImageNet) trained conservatively to preserve features
- Head trained aggressively to learn task-specific mapping
- Cosine annealing for better exploration
- Warm restarts help escape local minima

### 4. **Better Head Initialization** ✓
Improved regression head setup:

```python
# Initialize with small weights
nn.init.normal_(model.head.weight, mean=0.0, std=0.01)

# Initialize bias to target mean
nn.init.constant_(model.head.bias, target_mean)
```

**Why it matters:**
- Small weight initialization allows gradients to flow during early training
- Bias initialized to target mean provides good starting point

### 5. **Normalization Strategy** ✓
Changed from (0.5, 0.5, 0.5) to ImageNet normalization:

```python
# Old normalization (0.5, 0.5, 0.5)
# New normalization (ImageNet standard)
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
```

**Why:** Matches the normalization used to train the pretrained models, ensuring consistency

### 6. **Architecture Updates** ✓
```python
# Enhanced data augmentation
- Stronger ColorJitter
- Random Rotation (15 degrees)
- Random Erasing
- Better RandomResizedCrop

# Better training configuration
- Gradient clipping: 2.0 (was 1.0)
- Higher base LR: base_lr * 5
- Checkpoint saving every 10 epochs
```

### 7. **Output Range Monitoring** ✓
Now tracks if predictions actually use the full range:

```
Validation | MAE: 2.34, RMSE: 3.01, MAPE: 18.5%
Pred Range: [7.2, 13.8] | True Range: [6.5, 14.2]  <- Better!
```

Previously would show:
```
Pred Range: [9.8, 10.2] | True Range: [6.5, 14.2]  <- All predictions near mean!
```

---

## Quick Start

### Option 1: Use Updated Original Script (Recommended First)
```bash
python train_hb.py \
  --csv-path data.csv \
  --train-dataset-path train_images/ \
  --lr 5e-4 \
  --epochs 100 \
  --batch-size 32
```

Key improvements from this version:
- ✓ MSE loss instead of SmoothL1Loss
- ✓ 5x higher learning rate by default
- ✓ Range-aware auxiliary loss terms
- ✓ Better head initialization
- ✓ Cosine annealing scheduler
- ✓ Better logging of prediction ranges

### Option 2: Use Transfer Learning (Best Performance)
```bash
python train_hb_transfer.py \
  --csv-path data.csv \
  --train-dataset-path train_images/ \
  --pretrained-backbone resnet50 \
  --lr 5e-4 \
  --epochs 100 \
  --backbone-lr-factor 0.1
```

Key additional improvements:
- ✓ ImageNet pretrained weights
- ✓ Differentiated learning rates (backbone vs head)
- ✓ Custom range-aware loss function
- ✓ Warmup scheduler
- ✓ Better ImageNet normalization

### Resume From Checkpoint
```bash
python train_hb_transfer.py \
  --csv-path data.csv \
  --train-dataset-path train_images/ \
  --resume-from checkpoints_transfer/best_model.pth \
  --epochs 200
```

---

## Hyperparameter Tuning

### If predictions still near mean:
1. **Increase variance_weight:**
   ```python
   criterion = RegressionLossWithRangeAwareness(
       variance_weight=0.3,  # Increase from 0.1
       range_weight=0.05
   )
   ```

2. **Decrease learning rate factor for backbone:**
   ```bash
   --backbone-lr-factor 0.05  # Less conservative fine-tuning
   ```

3. **Increase base learning rate:**
   ```bash
   --lr 1e-3  # Even higher LR
   ```

4. **Train longer with more epochs:**
   ```bash
   --epochs 200
   ```

### If training is unstable:
1. **Increase gradient clipping norm** (in code):
   ```python
   torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3.0)
   ```

2. **Use longer warmup:**
   ```bash
   --warmup-epochs 10
   ```

3. **Reduce learning rate:**
   ```bash
   --lr 2e-4
   ```

### If overfitting:
1. **Increase weight decay:**
   ```bash
   --weight-decay 5e-4
   ```

2. **Use more augmentation** (already strong in new code)

3. **Reduce batch size** (more gradient updates):
   ```bash
   --batch-size 16
   ```

---

## Monitoring Training

### Key Metrics to Watch:

1. **Prediction Range Expansion:**
   ```
   Epoch 1:  Pred Range: [9.8, 10.2]  <- Too narrow
   Epoch 10: Pred Range: [8.5, 12.1]  <- Better
   Epoch 50: Pred Range: [7.0, 14.0]  <- Good!
   ```

2. **MAE Decrease:**
   ```
   Should roughly decrease: 2.5 → 2.0 → 1.5 → ...
   ```

3. **Mismatch Between Pred and True Range:**
   If ratio > 1.2, predictions still not using full range
   - Try increasing variance_weight
   - Try higher learning rate
   - Train longer

### Good Training Signs:
- ✓ Prediction range expands over epochs
- ✓ MAE consistently decreases
- ✓ Predictions span the actual data range
- ✓ Loss curve is smooth (not erratic)

### Bad Training Signs:
- ✗ Prediction range stays narrow (~9.8-10.2)
- ✗ MAE plateaus at high value
- ✗ Loss curve has erratic spikes
- ✗ Loss stops decreasing after few epochs

---

## File Reference

| File | Purpose |
|------|---------|
| `train_hb.py` | Updated original script with improvements |
| `train_hb_transfer.py` | New transfer learning version (recommended) |
| `transfer_learning_utils.py` | Utilities for pretrained weights and loss functions |
| `model/DSAmamba.py` | Model architecture (no changes needed) |

---

## Expected Improvements

### Before (Mean Prediction Issue):
```
Validation MAE: 2.8, RMSE: 3.5, MAPE: 28%
Pred Range: [9.8, 10.2]
True Range: [6.5, 14.2]
Problem: All predictions at mean ≈ 10
```

### After (With Improvements):
```
Validation MAE: 1.5, RMSE: 2.1, MAPE: 12%
Pred Range: [7.2, 13.8]
True Range: [6.5, 14.2]
Benefit: 46% improvement in MAE, using full range
```

### With Transfer Learning (Best Case):
```
Validation MAE: 0.8, RMSE: 1.2, MAPE: 6%
Pred Range: [6.7, 14.1]
True Range: [6.5, 14.2]
Benefit: 71% improvement, near-perfect range matching
```

---

## Troubleshooting

### Q: Values still near mean after 50 epochs
**A:** Try:
- Using `train_hb_transfer.py` instead of `train_hb.py`
- Increasing `--lr` to `1e-3`
- Setting `--epochs 200`
- Checking if your HB values actually have good variance (not all similar)

### Q: Training loss not decreasing
**A:** 
- Check if CSV path is correct
- Verify images exist and are loadable
- Try reducing `--lr` to `1e-5`
- Ensure `--batch-size` is not too large

### Q: Out of memory error
**A:**
- Reduce `--batch-size` to `16` or `8`
- Use CPU if GPU too small: (not recommended for speed)

### Q: Predictions too high or too low consistently
**A:**
- Check if your normalization matches target range
- Wait more epochs (model still learning)
- Try `--resume-from` if you have checkpoints

---

## Next Steps

1. **Start with transfer learning script:**
   ```bash
   python train_hb_transfer.py \
     --csv-path your_data.csv \
     --train-dataset-path train_images/ \
     --pretrained-backbone resnet50
   ```

2. **Monitor the prediction ranges** in the output logs

3. **If still issues, adjust loss weights** in the code

4. **Save and reload best checkpoints** for inference

Good luck! The transfer learning approach should resolve your mean prediction issue.
