# DSA-Mamba Training, Evaluation & Visualization Guide

## Overview

After training completes, the DSA-Mamba pipeline automatically:
1. ✓ Evaluates the best model on the validation set
2. ✓ Calculates comprehensive metrics (accuracy, AUC, precision, recall, F1, specificity)
3. ✓ Generates 7 detailed visualization plots
4. ✓ Saves results as JSON for further analysis

---

## Quick Start

### Basic Training Command (Your Current Setup)
```bash
python train.py \
  --train-dataset-path /kaggle/input/datasets/junaidgpu/cpanemic/cpanemic/CPanemic \
  --csv-path /kaggle/input/datasets/junaidgpu/cpanemic/cpanemic/Anemia_Data_Collection_Sheet.xlsx \
  --image-col "IMAGE_ID" \
  --hb-col "HB_LEVEL" \
  --hb-threshold 12.0 \
  --batch-size 4 \
  --epochs 5
```

**What happens:**
1. Model trains for 5 epochs
2. Best model is saved automatically
3. At the end, evaluation and plots are generated automatically
4. All results saved to `./eval_results/` and `./logs/`

---

## Output Files

### Evaluation Plots (in `./eval_results/`)

#### 1. **dsamamba_training_loss.png**
- **What it shows:** Training loss across all epochs
- **Why it matters:** Loss should decrease over time (indicates learning)
- **How to interpret:**
  - Downward trend = good (model is learning)
  - Flat/increasing = concern (model not improving)
  - Sharp drops = significant learning events

#### 2. **dsamamba_validation_metrics.png** (6 subplots)
Contains 6 key metrics over training:

- **Accuracy:** Overall correctness (% of correct predictions)
  - Good if: increasing over epochs
  - Concern if: stuck at previous epoch value

- **AUC (Area Under Curve):** Binary classification capability
  - Range: 0.5 (random) to 1.0 (perfect)
  - Good if: > 0.7
  - Current issue if: ≈ 0.58 (barely better than random)

- **Precision:** Of positive predictions, how many are correct?
  - Formula: TP / (TP + FP)
  - Good if: high (means few false alarms)
  - Current issue if: 0.359 (very low)

- **Sensitivity (Recall):** Of actual positive cases, how many did we catch?
  - Formula: TP / (TP + FN)
  - Good if: high (catches most anemic cases)
  - Current issue if: 0.5 (50% - frozen)

- **F1-Score:** Harmonic mean of precision and recall
  - Good if: high and increasing
  - Current issue if: frozen at 0.418

- **Specificity:** Of actual negative cases, how many did we correctly reject?
  - Formula: TN / (TN + FP)
  - Good if: high (correctly identifies non-anemic)
  - Current issue if: 0.5 (random guessing)

#### 3. **dsamamba_confusion_matrix.png**
```
Anemia Classification Confusion Matrix:
                Predicted
              Non-Anemic  Anemic
Actual  Non-Anemic    [TN]     [FP]
        Anemic        [FN]     [TP]
```
- **TN (True Negatives):** Correctly classified non-anemic
- **TP (True Positives):** Correctly classified anemic
- **FP (False Positives):** Incorrectly marked as anemic
- **FN (False Negatives):** Incorrectly marked as non-anemic
- **Good pattern:** High diagonal values (TN and TP)
- **Problem pattern:** Model predicts mostly one class

#### 4. **dsamamba_roc_curve.png**
- **What it shows:** Trade-off between True Positive Rate (sensitivity) vs False Positive Rate (1 - specificity)
- **Interpretation:**
  - Line close to top-left corner = excellent classifier
  - Line close to diagonal = random guessing
  - AUC in title = area under the curve (0.5 to 1.0)
- **Your current issue:** If AUC ≈ 0.58, model is barely discriminating between classes

#### 5. **dsamamba_class_distribution.png**
- **What it shows:** Count of anemic vs non-anemic in validation set
- **Why it matters:** Reveals class imbalance
- **Example issue:** If 80% non-anemic, 20% anemic → model may just predict all non-anemic to achieve 80% accuracy

#### 6. **dsamamba_metrics_summary.png**
- **What it shows:** Bar chart of all key metrics
- **Use:** Quick visual comparison of model performance
- **Good threshold:** Most bars should be > 0.7 for a well-performing model

### Results File (in `./eval_results/`)

#### **dsamamba_results.json**
Contains numerical results:
```json
{
  "accuracy": 0.7183,
  "precision": 0.359,
  "recall": 0.500,
  "f1_score": 0.418,
  "auc": 0.581,
  "sensitivity": 0.500,
  "specificity": 0.500,
  "confusion_matrix": [[TN, FP], [FN, TP]]
}
```

### Training Logs (in `./logs/`)

JSON files tracking metrics per epoch:
- `dsamamba_FETAL_loss.json` - Training loss
- `dsamamba_FETAL_acc.json` - Validation accuracy
- `dsamamba_FETAL_auc.json` - Validation AUC
- `dsamamba_FETAL_precision.json` - Validation precision
- `dsamamba_FETAL_sensitivity.json` - Validation sensitivity
- `dsamamba_FETAL_f1score.json` - Validation F1-score
- `dsamamba_FETAL_specificity.json` - Validation specificity

### Model Checkpoint (in `./pth_out/`)

- `dsamamba_FETAL_best.pth` - Best model weights

---

## Understanding Your Current Results

Your current metrics show a potential issue:

```
val_accuracy: 0.718 (frozen across epochs)
auc: 0.581 → 0.582 → 0.581 (oscillating)
precision: 0.359 (very low)
sensitivity: 0.500 (stuck at 50%)
f1: 0.418 (frozen)
specificity: 0.500 (stuck at 50%)
```

**Problem Analysis:**

1. **Model not learning** - Metrics frozen across epochs
2. **Class imbalance** - Likely predicting majority class only
3. **Poor discrimination** - AUC ≈ 0.58 (barely better than random 0.5)

**Solutions to try:**

### Fix 1: Check Class Distribution
```bash
python check_distribution.py
```
If output shows > 70% one class, you need class weighting.

### Fix 2: Add Class Weighting
The code will automatically detect imbalance and apply weights if you update the loss function:
```python
# Calculate class weights
class_counts = [non_anemic_count, anemic_count]
class_weights = torch.tensor([1.0, max(class_counts) / min(class_counts)])
loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

### Fix 3: Increase Learning Rate
Change in `train.py`:
```python
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Increased from 0.0001
```

### Fix 4: Add Data Augmentation
Enhance training data variation to improve generalization.

---

## Running the Full Pipeline

### Option 1: Direct Training Command
```bash
python train.py --train-dataset-path ... --csv-path ... --epochs 5
```

### Option 2: Using Pipeline Script (Recommended)
```bash
python run_training_pipeline.py
```
This script:
- Runs training with your Kaggle paths
- Shows all generated outputs
- Provides summary of results

---

## Plotting Without Training (Replotting)

If you want to regenerate plots from existing logs:

```python
from eval_and_plot import plot_training_history

plot_training_history(
    exp_name='dsamamba',
    log_dir='./logs',
    output_dir='./eval_results'
)
```

---

## Key Metrics Explained

| Metric | Formula | Range | Ideal | Meaning |
|--------|---------|-------|-------|---------|
| **Accuracy** | (TP + TN) / Total | 0-1 | 1.0 | Overall correctness |
| **Precision** | TP / (TP + FP) | 0-1 | 1.0 | Of predicted positives, % correct |
| **Recall/Sensitivity** | TP / (TP + FN) | 0-1 | 1.0 | Of actual positives, % caught |
| **Specificity** | TN / (TN + FP) | 0-1 | 1.0 | Of actual negatives, % correct |
| **F1-Score** | 2 × (Prec × Rec) / (Prec + Rec) | 0-1 | 1.0 | Balanced precision-recall |
| **AUC** | Area under ROC curve | 0.5-1.0 | 1.0 | Classification ability |

---

## Troubleshooting

### No plots generated after training?
- Check that `./eval_results/` directory was created
- Look for error messages in console output
- Ensure all dependencies installed: `pip install matplotlib seaborn scikit-learn`

### Metrics all frozen?
- Model may not be learning (check learning rate)
- Check class distribution
- Verify training loss is decreasing (`dsamamba_training_loss.png`)

### AUC ≈ 0.5?
- Model is predicting randomly
- Likely class imbalance issue
- Try increasing learning rate or adding class weights

### All plots generated but metrics don't make sense?
- Verify data annotation quality (check some images and HB labels manually)
- Ensure CSV/Excel mapping is correct
- Check if label threshold (HB_LEVEL < 12.0) makes sense for your data

---

## Next Steps After Evaluation

1. **Review the plots** - Focus on whether AUC and recall are improving
2. **Check class distribution** - Identify imbalance issues
3. **Adjust hyperparameters** if needed:
   - Increase `--epochs` for more training
   - Adjust `--batch-size` if GPU memory allows
   - Try different `--hb-threshold` values
4. **Retrain** - Run training again to see if metrics improve

---

## Questions?

- **Metrics not changing?** → Check class distribution, learning rate, and data quality
- **Very slow training?** → Check `train.py` for per-iteration time in progress bar (should be 20-30s/batch with batch_size=4)
- **Out of memory?** → Reduce `--batch-size` to 2 or 1

Good luck with your training! 🚀
