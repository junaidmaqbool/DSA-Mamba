#!/usr/bin/env python3
"""
HB Regression Training with Transfer Learning
Improved version with transfer learning, better loss functions, and hyperparameter tuning.

Usage:
    python train_hb_transfer.py --csv-path data.csv --train-dataset-path train_images/ \
                                --pretrained-backbone resnet50 --lr 5e-4 --epochs 100
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.DSAmamba import VSSM as dsamamba
from transfer_learning_utils import (
    load_pretrained_backbone_weights,
    initialize_regression_head,
    create_optimizer_with_lr_decay,
    create_warmup_scheduler,
    RegressionLossWithRangeAwareness
)


class HbRegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns (image, hb_value)."""
    
    def __init__(self, images_dir, mapping_df, image_col='image_name', hb_col='hb', 
                 transform=None, scaler=None):
        self.images_dir = images_dir
        if isinstance(mapping_df, str):
            if mapping_df.lower().endswith('.csv'):
                df = pd.read_csv(mapping_df)
            else:
                try:
                    df = pd.read_excel(mapping_df)
                except Exception:
                    df = pd.read_csv(mapping_df)
            self.df = df
        else:
            self.df = mapping_df.copy()

        self.image_col = image_col
        self.hb_col = hb_col
        self.transform = transform
        self.scaler = scaler
        self.samples = []
        skipped = 0

        for _, row in self.df.iterrows():
            img_name = str(row[self.image_col])
            img_base = os.path.basename(img_name)
            found = None
            
            for ext in IMG_EXTENSIONS:
                candidate = os.path.join(self.images_dir, 
                                        img_base if img_base.lower().endswith(ext) 
                                        else img_base + ext)
                if os.path.exists(candidate):
                    found = candidate
                    break

            if found is None:
                candidate2 = os.path.join(self.images_dir, img_name)
                if os.path.exists(candidate2):
                    found = candidate2

            if found is None and os.path.exists(img_name):
                found = img_name

            if found is None:
                skipped += 1
                continue

            try:
                hb_val = float(row[self.hb_col])
            except Exception:
                skipped += 1
                continue

            # Validate image
            try:
                with Image.open(found) as im:
                    im.verify()
            except Exception:
                skipped += 1
                continue

            self.samples.append((found, float(hb_val)))

        print(f"HbRegressionDataset: loaded {len(self.samples)} samples, skipped {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, hb_val = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            
            if self.scaler is not None:
                mean, std = self.scaler
                if std == 0 or np.isnan(std):
                    std = 1.0
                hb_norm = (hb_val - mean) / std
                return img, torch.tensor(hb_norm, dtype=torch.float32)
            return img, torch.tensor(hb_val, dtype=torch.float32)
        except Exception:
            img = Image.new('RGB', (224, 224))
            if self.transform is not None:
                img = self.transform(img)
            if self.scaler is not None:
                mean, std = self.scaler
                if std == 0 or np.isnan(std):
                    std = 1.0
                hb_norm = (hb_val - mean) / std
                return img, torch.tensor(hb_norm, dtype=torch.float32)
            return img, torch.tensor(hb_val, dtype=torch.float32)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, np.nan, y_true)
    res = np.abs((y_true - y_pred) / denom)
    res = res[~np.isnan(res)]
    return float(np.mean(res) * 100) if res.size > 0 else float('nan')


def get_args_parser():
    parser = argparse.ArgumentParser('HB Regression with Transfer Learning', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    parser.add_argument('--n-channels', default=3, type=int, help='Number of image channels')
    parser.add_argument('--train-dataset-path', default='train_images', type=str, help='Train dataset path')
    parser.add_argument('--val-dataset-path', default='val_images', type=str, help='Val dataset path')
    parser.add_argument('--csv-path', default=None, type=str, required=True, 
                       help='CSV with image filenames and HB values')
    parser.add_argument('--image-col', default='image_name', type=str, help='Image column name')
    parser.add_argument('--hb-col', default='hb', type=str, help='HB value column name')
    parser.add_argument('--val-split', default=0.2, type=float, help='Validation split ratio')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--save-dir', default='checkpoints_transfer', type=str, help='Save directory')
    parser.add_argument('--resume-from', default=None, type=str, help='Resume from checkpoint')
    parser.add_argument('--pretrained-backbone', default='resnet50', type=str, 
                       help='Pretrained backbone (resnet50, resnet101, densenet121, vit_b)')
    parser.add_argument('--backbone-lr-factor', default=0.1, type=float, 
                       help='LR factor for backbone vs head')
    parser.add_argument('--warmup-epochs', default=5, type=int, help='Warmup epochs')
    parser.add_argument('--loss-type', default='range', type=str,
                       choices=['mse','mae','huber','range'],
                       help='Type of loss to use (range uses custom RegressionLossWithRangeAwareness)')
    parser.add_argument('--balance-sampler', default=False, type=bool,
                       help='Use weighted sampler to balance training hb distribution')
    parser.add_argument('--freeze-backbone-epochs', default=0, type=int,
                       help='Number of epochs to freeze backbone weights at start')
    parser.add_argument('--early-stopping-patience', default=0, type=int,
                       help='Stop training if validation MAE does not improve for this many epochs (0 to disable)')
    parser.add_argument('--use-plateau-scheduler', default=False, type=bool,
                       help='Use ReduceLROnPlateau scheduler based on val MAE')
    return parser


def main():
    parser = argparse.ArgumentParser('HB Regression Transfer Learning', parents=[get_args_parser()])
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Data transforms with more augmentation for regularization
    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.7),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet norm
            transforms.RandomErasing(p=0.2),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet norm
        ])
    }

    if args.csv_path is None:
        raise RuntimeError('Please provide --csv-path')

    # Load mapping
    if args.csv_path.lower().endswith('.csv'):
        mapping_df = pd.read_csv(args.csv_path)
    else:
        try:
            mapping_df = pd.read_excel(args.csv_path)
        except Exception:
            mapping_df = pd.read_csv(args.csv_path)

    total_rows = len(mapping_df)
    if total_rows == 0:
        raise RuntimeError('Mapping file is empty')

    val_len = int(total_rows * float(args.val_split))
    train_len = total_rows - val_len
    if val_len == 0:
        val_len = max(1, total_rows // 10)
        train_len = total_rows - val_len

    train_df = mapping_df.iloc[:train_len].reset_index(drop=True)
    val_df = mapping_df.iloc[train_len:train_len+val_len].reset_index(drop=True)

    # Compute scaler
    hb_series = train_df[args.hb_col].astype(float)
    hb_mean = float(hb_series.mean())
    hb_std = float(hb_series.std())
    if hb_std == 0 or np.isnan(hb_std):
        hb_std = 1.0
    scaler = (hb_mean, hb_std)
    print(f"HB Statistics - Mean: {hb_mean:.3f}, Std: {hb_std:.3f}, "
          f"Range: [{hb_series.min():.1f}, {hb_series.max():.1f}]")

    # Create datasets
    train_dataset = HbRegressionDataset(args.train_dataset_path, train_df, 
                                       image_col=args.image_col, hb_col=args.hb_col, 
                                       transform=data_transform['train'], scaler=scaler)
    val_dataset = HbRegressionDataset(args.train_dataset_path, val_df, 
                                     image_col=args.image_col, hb_col=args.hb_col, 
                                     transform=data_transform['val'], scaler=scaler)

    # optionally balance by hb distribution using weighted sampler
    if args.balance_sampler:
        # compute weights inversely proportional to frequency in bins
        hb_vals = np.array([hb for _, hb in train_dataset.samples])
        if hb_vals.size > 0:
            num_bins = min(10, len(hb_vals))
            bins = np.linspace(hb_vals.min(), hb_vals.max(), num_bins + 1)
            bin_idx = np.digitize(hb_vals, bins)  # 1..num_bins
            counts = np.bincount(bin_idx, minlength=num_bins+1)
            # avoid division by zero
            counts = counts.astype(float)
            counts[counts == 0] = 1.0
            weights = 1.0 / counts[bin_idx]
            weights = torch.tensor(weights, dtype=torch.double)
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                      sampler=sampler, num_workers=4)
            print("Using weighted sampler to balance training distribution")
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                      shuffle=True, num_workers=4)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                                  shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, 
                                            shuffle=False, num_workers=4)

    # Create model
    net = dsamamba(in_chans=args.n_channels, num_classes=1)
    net.to(device)

    # optionally freeze backbone for a few epochs
    if args.freeze_backbone_epochs > 0:
        for name, param in net.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        print(f"Backbone frozen for first {args.freeze_backbone_epochs} epochs")

    # Pre-init lazy layers
    try:
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            _ = net(dummy)
    except Exception:
        pass

    # Load pretrained backbone weights
    print(f"\nLoading pretrained backbone: {args.pretrained_backbone}")
    net = load_pretrained_backbone_weights(net, args.pretrained_backbone, device)
    
    # Initialize regression head
    net = initialize_regression_head(net, target_mean=hb_mean, target_std=hb_std)

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume_from is not None and os.path.exists(args.resume_from):
        print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            net.load_state_dict(checkpoint['model_state'])
            start_epoch = checkpoint.get('epoch', 0) + 1
        else:
            net.load_state_dict(checkpoint)

    # Create optimizer with differentiated learning rates
    optimizer = create_optimizer_with_lr_decay(net, base_lr=args.lr, 
                                              weight_decay=args.weight_decay,
                                              backbone_lr_factor=args.backbone_lr_factor)

    # Create loss
    if args.loss_type == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_type == 'mae':
        criterion = nn.L1Loss()
    elif args.loss_type == 'huber':
        criterion = nn.SmoothL1Loss()
    elif args.loss_type == 'range':
        criterion = RegressionLossWithRangeAwareness(variance_weight=0.1, range_weight=0.05)
    else:
        criterion = nn.MSELoss()
    print(f"Using loss type: {args.loss_type}")

    # scheduler: either warmup+cosine or reduce-on-plateau
    if args.use_plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                               patience=5, verbose=True)
        print("Using ReduceLROnPlateau scheduler")
    else:
        scheduler = create_warmup_scheduler(optimizer, warmup_epochs=args.warmup_epochs, 
                                           total_epochs=args.epochs)

    # Training loop
    train_losses = []
    val_mae_list = []
    val_rmse_list = []
    val_mape_list = []
    best_val_mae = float('inf')

    print("\n" + "="*70)
    print("STARTING TRAINING WITH TRANSFER LEARNING")
    print("="*70 + "\n")

    no_improve_epochs = 0
    for epoch in range(start_epoch, args.epochs):
        net.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, file=sys.stdout, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # unfreeze backbone if needed
        if epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0:
            for name, param in net.named_parameters():
                if 'head' not in name:
                    param.requires_grad = True
            optimizer = create_optimizer_with_lr_decay(net, base_lr=args.lr, 
                                                      weight_decay=args.weight_decay,
                                                      backbone_lr_factor=args.backbone_lr_factor)
            # also rebuild scheduler to use the new optimizer
            if args.use_plateau_scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                                       patience=5, verbose=True)
            else:
                scheduler = create_warmup_scheduler(optimizer, warmup_epochs=args.warmup_epochs, 
                                                   total_epochs=args.epochs)
            print("Backbone unfrozen, optimizer and scheduler recreated")

        for images, hb_vals in pbar:
            images = images.to(device)
            hb_vals = hb_vals.view(-1, 1).to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, hb_vals)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += float(loss.item())
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        avg_loss = running_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

        # Validation
        net.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for images, hb_vals in val_loader:
                images = images.to(device)
                outputs = net(images)
                outputs = outputs.squeeze().cpu().numpy()
                trues = hb_vals.cpu().numpy().astype(float)
                
                preds_denorm = (outputs * hb_std) + hb_mean
                trues_denorm = (trues * hb_std) + hb_mean
                all_preds.extend(np.atleast_1d(preds_denorm).tolist())
                all_trues.extend(np.atleast_1d(trues_denorm).tolist())

        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        val_mae = mae(all_trues, all_preds)
        val_rmse = rmse(all_trues, all_preds)
        val_mape = mape(all_trues, all_preds)
        # r2 score
        tot_var = np.sum((all_trues - np.mean(all_trues))**2)
        res_var = np.sum((all_trues - all_preds)**2)
        val_r2 = 1.0 - (res_var / tot_var) if tot_var > 0 else float('nan')
        val_mae_list.append(val_mae)
        val_rmse_list.append(val_rmse)
        val_mape_list.append(val_mape)
        # store R2 separately if desired
        print(f"Validation | MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.2f}%, R2: {val_r2:.4f}")

        print(f"Pred Range: [{all_preds.min():.2f}, {all_preds.max():.2f}] | "
              f"True Range: [{all_trues.min():.2f}, {all_trues.max():.2f}]")

        # Step scheduler
        if args.use_plateau_scheduler:
            scheduler.step(val_mae)
        else:
            scheduler.step()

        # early stopping counter
        if args.early_stopping_patience > 0:
            if val_mae < best_val_mae - 1e-6:
                best_val_mae = val_mae
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            if no_improve_epochs >= args.early_stopping_patience:
                print(f"Early stopping triggered after {no_improve_epochs} epochs without improvement.")
                break
        else:
            if val_mae < best_val_mae:
                best_val_mae = val_mae

        # Save best model
        os.makedirs(args.save_dir, exist_ok=True)
        if val_mae <= best_val_mae + 1e-6:
            torch.save({
                'epoch': epoch,
                'model_state': net.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'hb_scaler': scaler
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✓ Best model saved! MAE: {val_mae:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': net.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Final evaluation
    print('\n' + "="*70)
    print('FINAL EVALUATION ON VALIDATION SET')
    print("="*70)
    
    net.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for path, hb_true in val_dataset.samples:
            img = Image.open(path).convert('RGB')
            img_t = data_transform['val'](img).unsqueeze(0).to(device)
            out_norm = net(img_t).squeeze().cpu().item()
            out_denorm = out_norm * hb_std + hb_mean
            preds.append(float(out_denorm))
            trues.append(float(hb_true))

    preds = np.array(preds)
    trues = np.array(trues)
    print(f"Samples: {len(preds)}")
    print(f"MAE: {mae(trues, preds):.4f}")
    print(f"RMSE: {rmse(trues, preds):.4f}")
    print(f"MAPE: {mape(trues, preds):.2f}%")
    # compute final R2
    tot_var = np.sum((trues - np.mean(trues))**2)
    res_var = np.sum((trues - preds)**2)
    r2_final = 1.0 - (res_var / tot_var) if tot_var > 0 else float('nan')
    print(f"R2: {r2_final:.4f}")
    print("="*70 + "\n")

    # Save plots
    os.makedirs('eval_results_transfer', exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('eval_results_transfer/train_loss.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    epochs = range(1, len(val_mae_list)+1)
    plt.plot(epochs, val_mae_list, label='MAE', marker='o')
    plt.plot(epochs, val_rmse_list, label='RMSE', marker='s')
    plt.plot(epochs, val_mape_list, label='MAPE (%)', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('eval_results_transfer/val_metrics.png')
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(trues, preds, alpha=0.6, s=50)
    mn, mx = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual HB')
    plt.ylabel('Predicted HB')
    plt.title('Predictions vs Actual (Validation)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('eval_results_transfer/pred_vs_actual.png')
    plt.close()

    print("Results and plots saved to eval_results_transfer/")


if __name__ == '__main__':
    main()
