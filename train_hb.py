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


class HbRegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns (image, hb_value) where hb_value is normalized if scaler provided.

    samples list stores tuples (path, hb_original_float) so we can later use original values for denorm.
    """
    def __init__(self, images_dir, mapping_df, image_col='image_name', hb_col='hb', transform=None, scaler=None):
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
                candidate = os.path.join(self.images_dir, img_base if img_base.lower().endswith(ext) else img_base + ext)
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

            # validate image can be opened
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
            # fallback zero image
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
    parser = argparse.ArgumentParser('Regression training for HB estimate', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--n-channels', default=3, type=int)
    parser.add_argument('--train-dataset-path', default='kvasir-dataset-v2/train', type=str)
    parser.add_argument('--val-dataset-path', default='kvasir-dataset-v2/val', type=str)
    parser.add_argument('--csv-path', default=None, type=str, help='mapping file with image filenames and hb column')
    parser.add_argument('--image-col', default='image_name', type=str)
    parser.add_argument('--hb-col', default='hb', type=str)
    parser.add_argument('--val-split', default=0.2, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save-dir', default='checkpoints', type=str)
    return parser


def main():
    parser = argparse.ArgumentParser('HB regression', parents=[get_args_parser()])
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.05)], p=0.7),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomErasing(p=0.2),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }

    if args.csv_path is None:
        raise RuntimeError('Please provide --csv-path mapping file with image filenames and hb values for regression')

    # read mapping
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

    # compute scaler from training hb values
    hb_series = train_df[args.hb_col].astype(float)
    hb_mean = float(hb_series.mean())
    hb_std = float(hb_series.std())
    if hb_std == 0 or np.isnan(hb_std):
        hb_std = 1.0
    scaler = (hb_mean, hb_std)
    print(f"Using HB scaler mean={hb_mean:.3f}, std={hb_std:.3f}")

    train_dataset = HbRegressionDataset(args.train_dataset_path, train_df, image_col=args.image_col,
                                        hb_col=args.hb_col, transform=data_transform['train'], scaler=scaler)
    val_dataset = HbRegressionDataset(args.train_dataset_path, val_df, image_col=args.image_col,
                                      hb_col=args.hb_col, transform=data_transform['val'], scaler=scaler)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    net = dsamamba(in_chans=args.n_channels, num_classes=1)
    net.to(device)

    # Pre-init lazy
    try:
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            _ = net(dummy)
    except Exception:
        pass

    # Use Huber (Smooth L1) loss for robustness to outliers
    criterion = nn.SmoothL1Loss()
    # AdamW with weight decay for better generalization
    optimizer = optim.AdamW(net.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    # Reduce LR when validation plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_losses = []
    val_mae_list = []
    val_rmse_list = []
    val_mape_list = []

    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, file=sys.stdout, desc=f"Train Epoch {epoch+1}/{args.epochs}")
        for images, hb_vals in pbar:
            images = images.to(device)
            hb_vals = hb_vals.view(-1, 1).to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, hb_vals)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        avg_loss = running_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss: {avg_loss:.4f}")

        # validation metrics (denormalize for reporting)
        net.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for images, hb_vals in val_loader:
                images = images.to(device)
                outputs = net(images)
                outputs = outputs.squeeze().cpu().numpy()
                trues = hb_vals.cpu().numpy().astype(float)
                # denormalize
                preds_denorm = (outputs * hb_std) + hb_mean
                trues_denorm = (trues * hb_std) + hb_mean
                all_preds.extend(np.atleast_1d(preds_denorm).tolist())
                all_trues.extend(np.atleast_1d(trues_denorm).tolist())

        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        val_mae = mae(all_trues, all_preds)
        val_rmse = rmse(all_trues, all_preds)
        val_mape = mape(all_trues, all_preds)
        val_mae_list.append(val_mae)
        val_rmse_list.append(val_rmse)
        val_mape_list.append(val_mape)
        print(f"Validation MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE(%): {val_mape:.2f}")
        # step scheduler on validation MAE
        try:
            scheduler.step(val_mae)
        except Exception:
            pass

        # Save best model by MAE
        os.makedirs(args.save_dir, exist_ok=True)
        if epoch == 0:
            best_val_mae = val_mae
            torch.save({'epoch': epoch, 'model_state': net.state_dict(), 'optimizer_state': optimizer.state_dict()},
                       os.path.join(args.save_dir, 'best_model.pth'))
        else:
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save({'epoch': epoch, 'model_state': net.state_dict(), 'optimizer_state': optimizer.state_dict()},
                           os.path.join(args.save_dir, 'best_model.pth'))

    # Final evaluation on validation set using stored sample paths (denormalize predictions)
    print('\nFinal evaluation on validation set:')
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
    print(f"Samples evaluated: {len(preds)}")
    print(f"MAE: {mae(trues, preds):.4f}")
    print(f"RMSE: {rmse(trues, preds):.4f}")
    print(f"MAPE(%): {mape(trues, preds):.2f}")

    # Print actual vs predicted for each example (path, true, pred)
    print('\nActual vs Predicted (validation set):')
    for (p, hb_val), pred in zip(val_dataset.samples, preds.tolist()):
        print(f"{os.path.basename(p)}\ttrue: {hb_val:.2f}\tpred: {pred:.2f}")

    # Create plots directory and save graphs
    os.makedirs('eval_results', exist_ok=True)

    try:
        plt.figure()
        plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss (MSE on normalized HB)')
        plt.title('Training Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('eval_results/train_loss.png')
        plt.close()
    except Exception as e:
        print('Warning: failed to save train loss plot:', e)

    try:
        plt.figure()
        epochs = range(1, len(val_mae_list)+1)
        plt.plot(epochs, val_mae_list, label='MAE')
        plt.plot(epochs, val_rmse_list, label='RMSE')
        plt.plot(epochs, val_mape_list, label='MAPE (%)')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Validation Metrics (denormalized)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('eval_results/val_metrics.png')
        plt.close()
    except Exception as e:
        print('Warning: failed to save validation metrics plot:', e)

    try:
        plt.figure(figsize=(6,6))
        plt.scatter(trues, preds, alpha=0.6)
        if len(trues) > 0 and len(preds) > 0:
            mn = min(min(trues), min(preds))
            mx = max(max(trues), max(preds))
            plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel('Actual HB')
        plt.ylabel('Predicted HB')
        plt.title('Predicted vs Actual HB (validation)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('eval_results/pred_vs_actual.png')
        plt.close()
        print('\nSaved plots to eval_results/')
    except Exception as e:
        print('Warning: failed to save pred vs actual plot:', e)


if __name__ == '__main__':
    main()
