import os
import sys
import json
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.datasets.folder import IMG_EXTENSIONS
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

from model.DSAmamba import VSSM as dsamamba


class HbRegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns (image, hb_value_float) from a mapping file or dataframe."""
    def __init__(self, images_dir, mapping_df, image_col='image_name', hb_col='hb', transform=None):
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
            return img, torch.tensor(hb_val, dtype=torch.float32)
        except Exception as e:
            # fallback zero image
            img = Image.new('RGB', (224, 224))
            if self.transform is not None:
                img = self.transform(img)
            return img, torch.tensor(hb_val, dtype=torch.float32)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    # avoid division by zero: ignore zero true values in denominator
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
    return parser


def main():
    parser = argparse.ArgumentParser('HB regression', parents=[get_args_parser()])
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }

    if args.csv_path is None:
        # fallback to ImageFolder regression is not supported, so require csv mapping
        raise RuntimeError('Please provide --csv-path mapping file with image filenames and hb values for regression')

    # read mapping
    if args.csv_path.lower().endswith('.csv'):
        mapping_df = pd.read_csv(args.csv_path)
    else:
        try:
            mapping_df = pd.read_excel(args.csv_path)
        except Exception:
            mapping_df = pd.read_csv(args.csv_path)

    # split mapping
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

    train_dataset = HbRegressionDataset(args.train_dataset_path, train_df, image_col=args.image_col,
                                        hb_col=args.hb_col, transform=data_transform['train'])
    val_dataset = HbRegressionDataset(args.train_dataset_path, val_df, image_col=args.image_col,
                                      hb_col=args.hb_col, transform=data_transform['val'])

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

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=float(args.lr))

    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, file=sys.stdout, desc=f"Train Epoch {epoch+1}/{args.epochs}")
        for images, hb_vals in pbar:
            images = images.to(device)
            hb_vals = hb_vals.view(-1, 1).to(device)

            optimizer.zero_grad()
            outputs = net(images)
            # outputs shape (B,1)
            loss = criterion(outputs, hb_vals)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{args.epochs} train_loss: {avg_loss:.4f}")

        # validation metrics
        net.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for images, hb_vals in val_loader:
                images = images.to(device)
                outputs = net(images)
                outputs = outputs.squeeze().cpu().numpy()
                trues = hb_vals.cpu().numpy().astype(float)
                all_preds.extend(outputs.tolist())
                all_trues.extend(trues.tolist())

        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        val_mae = mae(all_trues, all_preds)
        val_rmse = rmse(all_trues, all_preds)
        val_mape = mape(all_trues, all_preds)
        print(f"Validation MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE(%): {val_mape:.2f}")

    # Final evaluation and printouts on validation (test) set
    print('\nFinal evaluation on validation set:')
    net.eval()
    preds = []
    trues = []
    paths = []
    with torch.no_grad():
        for img, hb in val_dataset:
            x = img.unsqueeze(0).to(device)
            out = net(x).squeeze().cpu().item()
            preds.append(float(out))
            trues.append(float(hb))
            # find path in samples stored
            # val_dataset.samples corresponds order
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


if __name__ == '__main__':
    main()
