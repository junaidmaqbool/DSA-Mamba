"""
Evaluation and plotting module for DSA-Mamba training.
Runs after training completes to generate test results and visualizations.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                             roc_auc_score, precision_score, recall_score, 
                             f1_score)
from tqdm import tqdm
import sys

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_metrics_from_logs(exp_name, env_name, log_dir='./logs'):
    """Load metrics from JSON log files."""
    log_file = os.path.join(log_dir, f"{exp_name}_{env_name}.json")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return json.load(f)
    return []


def plot_training_loss(exp_name, log_dir='./logs'):
    """Plot training loss over epochs."""
    loss_metrics = load_metrics_from_logs(exp_name, 'loss', log_dir)
    
    if not loss_metrics:
        print(f"No loss metrics found for {exp_name}")
        return
    
    epochs = [m['step'] for m in loss_metrics]
    losses = [m['score'] for m in loss_metrics]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=12, fontweight='bold')
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(log_dir, f"{exp_name}_training_loss.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()


def plot_validation_metrics(exp_name, log_dir='./logs'):
    """Plot all validation metrics over epochs."""
    metrics = {
        'acc': 'Accuracy',
        'auc': 'AUC',
        'precision': 'Precision',
        'sensitivity': 'Sensitivity (Recall)',
        'f1score': 'F1-Score',
        'specificity': 'Specificity'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name) in enumerate(metrics.items()):
        data = load_metrics_from_logs(exp_name, metric_key, log_dir)
        
        if data:
            epochs = [m['step'] for m in data]
            scores = [m['score'] for m in data]
            
            axes[idx].plot(epochs, scores, marker='o', linewidth=2, markersize=8, color='#4ECDC4')
            axes[idx].fill_between(epochs, scores, alpha=0.2, color='#4ECDC4')
            axes[idx].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel(metric_name, fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{metric_name} Over Epochs', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([0, 1.0])
    
    plt.tight_layout()
    plot_path = os.path.join(log_dir, f"{exp_name}_validation_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_path}")
    plt.close()


def evaluate_and_plot(model, val_loader, model_path, device, exp_name='dsamamba', 
                      log_dir='./logs', output_dir='./eval_results'):
    """
    Evaluate model on validation set and create detailed plots.
    
    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        model_path: Path to saved model weights
        device: Device (cuda/cpu)
        exp_name: Experiment name
        log_dir: Directory where logs are saved
        output_dir: Directory to save evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load model weights
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model path not found: {model_path}")
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating on validation set...")
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout, desc="Evaluating")
        for val_images, val_labels in val_bar:
            val_images = val_images.to(device, non_blocking=True)
            val_labels = val_labels.view(-1).long()
            
            outputs = model(val_images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(val_labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Diagnostic: Check if model is predicting only one class
    unique_preds = np.unique(all_preds)
    if len(unique_preds) == 1:
        print("\n" + "⚠️ "*25)
        print("WARNING: MODEL IS PREDICTING ONLY ONE CLASS!")
        print(f"All {len(all_preds)} predictions are class {unique_preds[0]}")
        print(f"Actual labels distribution: {np.bincount(all_labels.astype(int))}")
        print("\nThis usually means:")
        print("  1. CLASS IMBALANCE - Too many samples of one class")
        print("  2. POOR LEARNING - Learning rate too low or model not training")
        print("  3. DATA QUALITY - Check if labels are correct")
        print("\nRecommended fixes:")
        print("  • Check class distribution with check_distribution.py")
        print("  • Try increasing learning rate (0.0001 → 0.001)")
        print("  • Add class weights to CrossEntropyLoss")
        print("  • Validate data annotation quality")
        print("⚠️ "*25 + "\n")
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Binary classification specific metrics
    if len(np.unique(all_labels)) == 2:
        try:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            roc_auc = auc(fpr, tpr)
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            auc_score = 0.0
            
        # Specificity (True Negative Rate) - force 2x2 matrix with labels=[0, 1]
        try:
            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                specificity = 0.0
        except Exception as e:
            print(f"Warning: Could not calculate confusion matrix: {e}")
            specificity = 0.0
            
        sensitivity = recall  # For binary, recall = sensitivity
    else:
        try:
            auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"Warning: Could not calculate multiclass AUC: {e}")
            auc_score = 0.0
        specificity = None
        sensitivity = recall
    
    # Print evaluation results
    print("\n" + "="*50)
    print("VALIDATION SET EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"AUC:         {auc_score:.4f}")
    if specificity is not None:
        print(f"Specificity: {specificity:.4f}")
    print("="*50 + "\n")
    
    # Plot 1: Confusion Matrix
    try:
        # Try to make 2x2 matrix for binary classification
        if len(np.unique(all_labels)) == 2:
            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        else:
            cm = confusion_matrix(all_labels, all_preds)
    except Exception as e:
        print(f"Warning: Could not compute confusion matrix: {e}")
        cm = np.array([[1, 0], [0, 1]])  # Dummy matrix
    
    plt.figure(figsize=(8, 6))
    
    # Build heatmap with appropriate labels
    heatmap_kwargs = {'annot': True, 'fmt': 'd', 'cmap': 'Blues', 'cbar': True}
    if len(np.unique(all_labels)) == 2:
        heatmap_kwargs['xticklabels'] = ['Non-Anemic', 'Anemic']
        heatmap_kwargs['yticklabels'] = ['Non-Anemic', 'Anemic']
    
    sns.heatmap(cm, **heatmap_kwargs)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{exp_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {cm_path}")
    plt.close()
    
    # Plot 2: ROC Curve (for binary classification)
    if len(np.unique(all_labels)) == 2:
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='#FF6B6B', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            plt.title('ROC Curve - Validation Set', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            roc_path = os.path.join(output_dir, f"{exp_name}_roc_curve.png")
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {roc_path}")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create ROC curve: {e}")
            print("  (This may happen if model predicts only one class)")
    
    # Plot 3: Class Distribution
    unique, counts = np.unique(all_labels, return_counts=True)
    class_names = ['Non-Anemic', 'Anemic'] if len(unique) == 2 else [f'Class {i}' for i in unique]
    
    plt.figure(figsize=(8, 6))
    colors = ['#4ECDC4', '#FF6B6B']
    bars = plt.bar(class_names[:len(unique)], counts, color=colors[:len(unique)], alpha=0.7, edgecolor='black')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Validation Set Class Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    dist_path = os.path.join(output_dir, f"{exp_name}_class_distribution.png")
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {dist_path}")
    plt.close()
    
    # Plot 4: Metrics Summary Bar Chart
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': auc_score
    }
    if specificity is not None:
        metrics_dict['Specificity'] = specificity
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_dict.keys(), metrics_dict.values(), color='#4ECDC4', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Evaluation Metrics Summary', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, f"{exp_name}_metrics_summary.png")
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {metrics_path}")
    plt.close()
    
    # Save evaluation results as JSON
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc': float(auc_score),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity) if specificity is not None else None,
        'confusion_matrix': cm.tolist()
    }
    
    results_file = os.path.join(output_dir, f"{exp_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {results_file}")
    
    return results


def plot_training_history(exp_name, log_dir='./logs', output_dir='./eval_results'):
    """Plot all training history metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training loss
    plot_training_loss(exp_name, log_dir)
    
    # Plot validation metrics
    plot_validation_metrics(exp_name, log_dir)
    
    print(f"\n✓ All training history plots saved to {log_dir}")


if __name__ == "__main__":
    print("Evaluation and plotting module loaded successfully.")
    print("Use: from eval_and_plot import evaluate_and_plot, plot_training_history")
