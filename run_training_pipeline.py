#!/usr/bin/env python
"""
Complete Training, Evaluation, and Plotting Pipeline for DSA-Mamba
Run this script to train the model and automatically generate evaluation plots.
"""

import subprocess
import sys
import os

def main():
    print("\n" + "="*70)
    print("DSA-MAMBA TRAINING WITH AUTOMATIC EVALUATION AND PLOTTING")
    print("="*70 + "\n")
    
    # Training command - Kaggle example
    training_cmd = [
        "python", "train.py",
        "--train-dataset-path", "/kaggle/input/datasets/junaidgpu/cpanemic/cpanemic/CPanemic",
        "--csv-path", "/kaggle/input/datasets/junaidgpu/cpanemic/cpanemic/Anemia_Data_Collection_Sheet.xlsx",
        "--image-col", "IMAGE_ID",
        "--hb-col", "HB_LEVEL",
        "--hb-threshold", "12.0",
        "--batch-size", "4",
        "--epochs", "5"
    ]
    
    print("STEP 1: Running Training Pipeline")
    print("-" * 70)
    print(f"Command: {' '.join(training_cmd)}\n")
    
    try:
        result = subprocess.run(training_cmd, check=True)
        print("\n✓ Training completed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error running training: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    print("\nGenerated Output Files:")
    print("-" * 70)
    
    # List generated files
    output_dirs = ['./logs', './eval_results', './pth_out']
    for directory in output_dirs:
        if os.path.exists(directory):
            files = os.listdir(directory)
            if files:
                print(f"\n{directory}:")
                for f in sorted(files):
                    file_path = os.path.join(directory, f)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path) / 1024  # KB
                        print(f"  ✓ {f} ({size:.1f} KB)")
    
    print("\n" + "="*70)
    print("PLOTS AND METRICS")
    print("="*70)
    print("""
Generated Visualizations:
  1. dsamamba_training_loss.png
     - Shows training loss over all epochs
     - Lower loss indicates better training
     
  2. dsamamba_validation_metrics.png (6 subplots)
     - Accuracy: Overall correctness
     - AUC: Model's ability to discriminate between classes
     - Precision: Accuracy of positive predictions
     - Sensitivity/Recall: Coverage of positive class
     - F1-Score: Harmonic mean of precision and recall
     - Specificity: Coverage of negative class
     
  3. dsamamba_confusion_matrix.png
     - Shows True Positives, False Positives, True Negatives, False Negatives
     - Diagonal values are correct predictions
     
  4. dsamamba_roc_curve.png
     - Receiver Operating Characteristic curve
     - Area Under Curve (AUC) indicates discriminative ability
     - Closer to top-left (1.0) = better classifier
     
  5. dsamamba_class_distribution.png
     - Distribution of anemic vs non-anemic samples in validation set
     
  6. dsamamba_metrics_summary.png
     - Bar chart of all key evaluation metrics
     
  7. dsamamba_results.json
     - Numerical results for all metrics

All plots are saved to: ./eval_results/
All training logs are saved to: ./logs/
Best model checkpoint saved to: ./pth_out/dsamamba_FETAL_best.pth
""")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
