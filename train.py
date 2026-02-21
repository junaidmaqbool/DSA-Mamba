import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.datasets.folder import IMG_EXTENSIONS
import torch.optim as optim
from tqdm import tqdm

from model.DSAmamba import VSSM as dsamamba  # import model


import rl_plotter
from rl_plotter import logger
import argparse
import pandas as pd
from PIL import Image
import medmnist
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Import medmnist INFO and Evaluator conditionally 
# (only needed if using medmnist datasets via --medmnist flag)
try:
    from medmnist import INFO, Evaluator
except ImportError:
    INFO = {}
    Evaluator = None


class HbImageDataset(torch.utils.data.Dataset):
    """Custom dataset that maps image files to HB values (from Excel/CSV) and produces binary labels.

    Expects a DataFrame or path with at least filename and hb columns. Labels: 1=anemic (hb < threshold), 0=non-anemic.
    Automatically validates and skips corrupted or missing images.
    """
    def __init__(self, images_dir, mapping_df, image_col='image_name', hb_col='hb', transform=None, hb_threshold=12.0):
        self.images_dir = images_dir
        if isinstance(mapping_df, str):
            # try csv first, then excel
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
        self.hb_threshold = float(hb_threshold)

        # build samples list (full paths and labels), with validation
        self.samples = []
        skipped_count = 0
        valid_count = 0
        
        for idx, (_, row) in enumerate(self.df.iterrows()):
            skip_reason = None
            
            # Find image file
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

            if found is None:
                if os.path.exists(img_name):
                    found = img_name
            
            if found is None:
                skip_reason = "file_not_found"
            else:
                # Validate HB value
                try:
                    hb_val = float(row[self.hb_col])
                except Exception as e:
                    skip_reason = f"invalid_hb_value ({str(e)})"
                else:
                    # Try to open and validate the image
                    try:
                        with Image.open(found) as img:
                            img.verify()  # Verify the image is not corrupted
                        # Reopen to check if it can be converted to RGB
                        with Image.open(found) as img:
                            _ = img.convert('RGB')
                        
                        # All checks passed - add to samples
                        label = 1 if hb_val < self.hb_threshold else 0
                        self.samples.append((found, label))
                        valid_count += 1
                        
                    except Exception as e:
                        skip_reason = f"corrupted_image ({str(e)[:30]})"
            
            if skip_reason:
                skipped_count += 1
        
        print(f"HbImageDataset loaded: {valid_count} valid images, {skipped_count} skipped")
        if skipped_count > 0:
            print(f"  Reason for skipped images: check logs above for details")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Warning: Failed to load image at {path}: {e}. Returning a zero tensor instead.")
            # Return a zero tensor as fallback to prevent training interruption
            if self.transform is not None:
                # Try to infer shape from an empty transform
                try:
                    dummy_img = Image.new('RGB', (224, 224))
                    img = self.transform(dummy_img)
                except Exception:
                    img = torch.zeros(3, 224, 224)
            else:
                img = torch.zeros(3, 224, 224)
            return img, label



def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--num-classes', default=6, type=int)
    parser.add_argument('--n-channels', default=3, type=int)
    # parser.add_argument('--model-name', default='dsamamba', type=str)
    parser.add_argument('--num-works', default=4, type=int)
    parser.add_argument('--env-name', default='FETAL', type=str)
    parser.add_argument('--train-dataset-path', default='kvasir-dataset-v2/train',
                        type=str)
    parser.add_argument('--val-dataset-path', default='kvasir-dataset-v2/val', type=str)
    parser.add_argument('--csv-path', default=None, type=str,
                        help='Path to CSV/Excel mapping file with image filenames and HB values')
    parser.add_argument('--image-col', default='image_name', type=str,
                        help='Column name in mapping file containing image filenames')
    parser.add_argument('--hb-col', default='hb', type=str,
                        help='Column name in mapping file containing HB values')
    parser.add_argument('--hb-threshold', default=12.0, type=float,
                        help='HB threshold to decide anemia (hb < threshold => anemic)')
    parser.add_argument('--val-split', default=0.2, type=float,
                        help='If `val-dataset-path` does not exist, split `train-dataset-path` by this fraction for validation')
    parser.add_argument('--medmnist', default=False, type=bool)
    parser.add_argument('--medmnist-download', default=False, type=bool)
    parser.add_argument('--medmnist-choice', default='octmnist', type=str,
                        choices=['pathmnist', 'chestmnist', 'dermamnist', 'octmnist',
                                 'pneumoniamnist', 'retinamnist', 'breastmnist', 'organmnist_axial',
                                 'organmnist_coronal', 'organmnist_sagittal', 'organamnist', 'organcmnist',
                                 'organsmnist'])

    return parser


def main():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    model_name = 'dsamamba'
    print(f'Current model: {model_name}')
    print(f'Current datasets: {args.env_name}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if args.medmnist is False:
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            "val": transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
        # If a separate validation folder exists use it, otherwise split the provided dataset root dynamically
        train_root = args.train_dataset_path
        val_root = args.val_dataset_path

        # If a mapping CSV/Excel is provided, build datasets from it (binary anemia labels)
        if args.csv_path is not None:
            print(f"Loading mapping from {args.csv_path}")
            # read mapping
            if args.csv_path.lower().endswith('.csv'):
                mapping_df = pd.read_csv(args.csv_path)
            else:
                try:
                    mapping_df = pd.read_excel(args.csv_path)
                except Exception:
                    mapping_df = pd.read_csv(args.csv_path)

            # If val_root exists and points to a folder, use it for validation mapping if a separate file exists
            # Otherwise split mapping into train/val
            if os.path.isdir(val_root):
                train_dataset = HbImageDataset(train_root, mapping_df, image_col=args.image_col,
                                               hb_col=args.hb_col, transform=data_transform['train'],
                                               hb_threshold=args.hb_threshold)
                validate_dataset = datasets.ImageFolder(root=val_root, transform=data_transform['val'])
            else:
                # split mapping rows
                total_rows = len(mapping_df)
                if total_rows == 0:
                    raise RuntimeError(f"No entries found in mapping {args.csv_path}")
                val_frac = float(args.val_split)
                if not (0.0 <= val_frac < 1.0):
                    raise ValueError("--val-split must be in [0.0, 1.0)")
                val_len = int(total_rows * val_frac)
                train_len = total_rows - val_len
                if val_len == 0:
                    val_len = max(1, total_rows // 10)
                    train_len = total_rows - val_len

                train_df = mapping_df.iloc[:train_len].reset_index(drop=True)
                val_df = mapping_df.iloc[train_len:train_len+val_len].reset_index(drop=True)

                train_dataset = HbImageDataset(train_root, train_df, image_col=args.image_col,
                                               hb_col=args.hb_col, transform=data_transform['train'],
                                               hb_threshold=args.hb_threshold)
                validate_dataset = HbImageDataset(train_root, val_df, image_col=args.image_col,
                                                  hb_col=args.hb_col, transform=data_transform['val'],
                                                  hb_threshold=args.hb_threshold)

            # set num_classes for binary task
            args.num_classes = 2
            # create class mapping
            cla_dict = {0: 'non_anemic', 1: 'anemic'}
            with open('class_indices.json', 'w') as json_file:
                json.dump(cla_dict, json_file, indent=4)

            train_num = len(train_dataset)
            val_num = len(validate_dataset)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.num_works)
            validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=args.num_works)
            print(f"using {train_num} images for training, {val_num} images for validation (from mapping).")
            # Skip the rest of folder-based handling
        else:

            if os.path.isdir(val_root):
                train_dataset = datasets.ImageFolder(root=train_root, transform=data_transform["train"])
                validate_dataset = datasets.ImageFolder(root=val_root, transform=data_transform["val"])

                train_num = len(train_dataset)
                val_num = len(validate_dataset)

                flower_list = train_dataset.class_to_idx
                cla_dict = dict((val, key) for key, val in flower_list.items())
                json_str = json.dumps(cla_dict, indent=4)
                with open('class_indices.json', 'w') as json_file:
                    json_file.write(json_str)

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.num_works)
                validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False,
                                                            num_workers=args.num_works)
                print("using {} images for training, {} images for validation.".format(train_num, val_num))
            else:
                # Single root provided: use ImageFolder and split into train/val GG 
                full_dataset = datasets.ImageFolder(root=train_root, transform=data_transform["train"])
                total = len(full_dataset)
                if total == 0:
                    raise RuntimeError(f"No images found in {train_root}")

                val_frac = float(args.val_split)
                if not (0.0 <= val_frac < 1.0):
                    raise ValueError("--val-split must be in [0.0, 1.0)")

                val_len = int(total * val_frac)
                train_len = total - val_len
                if val_len == 0:
                    val_len = max(1, total // 10)
                    train_len = total - val_len

                train_dataset, validate_dataset = torch.utils.data.random_split(full_dataset, [train_len, val_len])

                # save class mapping using the underlying dataset
                flower_list = full_dataset.class_to_idx
                cla_dict = dict((val, key) for key, val in flower_list.items())
                with open('class_indices.json', 'w') as json_file:
                    json.dump(cla_dict, json_file, indent=4)

                train_num = train_len
                val_num = val_len

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.num_works)
                validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False,
                                                            num_workers=args.num_works)
                print(f"Dynamically split {total} images -> {train_num} train, {val_num} val (val_split={val_frac})")
    else:
        print('use medmnist datasets')
        if not INFO:
            raise ImportError("medmnist is not properly installed. Please install it with: pip install medmnist")
        info = INFO[args.medmnist_choice]
        # task = info['task']
        args.n_channels = info['n_channels']
        args.num_classes = len(info['label'])
        # print('len labels {}'.format(len(info['label'])))
        DataClass = getattr(medmnist, info['python_class'])

        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # Load datasets
        custom_root = 'kvasir-dataset-v2'

        train_dataset = DataClass(split='train', transform=data_transform, download=args.medmnist_download,
                                  root=custom_root)
        train_num = len(train_dataset)
        val_dataset = DataClass(split='val', transform=data_transform, download=args.medmnist_download,
                                root=custom_root)
        val_num = len(val_dataset)
        # test_dataset = DataClass(split='test', transform=data_transform, download=args.medmnist_download)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_works)
        validate_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=args.num_works)
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_works)
        print("using {} images for training, {} images for validation.".format(train_num, val_num))

    print(f"Instantiating {model_name} model...")
    net = dsamamba(in_chans=args.n_channels, num_classes=args.num_classes)

    best_acc = 0.0
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 0.0001

    acc_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_acc')
    auc_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_auc')
    pre_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_precision')
    sen_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_sensitivity')
    f1s_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_f1score')
    spe_logger = rl_plotter.logger.Logger(exp_name=model_name, env_name=args.env_name + '_specificity')


    save_path = f'./pth_out/{model_name}_FETAL_best.pth'
    os.makedirs('./pth_out', exist_ok=True)

    train_steps = len(train_loader)
    for epoch in range(args.epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for steps, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()

            labels = labels.squeeze().long().to(device)
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     args.epochs,
                                                                     loss)

        # validate
        val_accurate, auc, precision, sensitivity, f1, specificity = calculate_metrics(net, validate_loader, val_num,
                                                                                       device=device)

        print(
            '[epoch %d] train_loss: %.3f  val_accuracy: %.3f, auc: %.3f, precision: %.3f, sensitivity: %.3f, f1: %.3f, specificity: %.3f' %
            (epoch + 1, running_loss / train_steps, val_accurate, auc, precision, sensitivity, f1, specificity))

        acc_logger.update(score=[val_accurate], total_steps=epoch + 1)
        auc_logger.update(score=[auc], total_steps=epoch + 1)
        pre_logger.update(score=[precision], total_steps=epoch + 1)
        sen_logger.update(score=[sensitivity], total_steps=epoch + 1)
        f1s_logger.update(score=[f1], total_steps=epoch + 1)
        spe_logger.update(score=[specificity], total_steps=epoch + 1)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print(f"New best model saved to {save_path} with accuracy: {best_acc:.4f}")

    print('Finished Training')


# Function to calculate metrics
def calculate_metrics(model, validate_loader, val_num, device):
    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    pred_scores = []
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout, desc="Validating")
        for val_data in val_bar:
            val_images, val_labels = val_data
            val_labels = val_labels.squeeze().to(device)

            outputs = model(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]

            acc += torch.eq(predict_y, val_labels).sum().item()
            probabilities = torch.softmax(outputs, dim=1)

            all_labels.extend(val_labels.cpu().numpy())
            all_predictions.extend(predict_y.cpu().numpy())
            pred_scores.extend(probabilities.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    pred_scores = np.array(pred_scores)

    val_accurate = acc / val_num

    # Check if it's a multi-class or binary problem for AUC
    if pred_scores.shape[1] > 2:
        auc = roc_auc_score(all_labels, pred_scores, multi_class='ovr', average='macro')
    else:  # Binary case
        auc = roc_auc_score(all_labels, pred_scores[:, 1])

    # Calculate Precision, Recall (Sensitivity), and F1-score
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)  # Sensitivity
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    cm = confusion_matrix(all_labels, all_predictions)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)


    specificity_per_class = tn / (tn + fp)
    specificity_per_class[np.isnan(specificity_per_class)] = 0
    specificity = np.mean(specificity_per_class)

    return val_accurate, auc, precision, recall, f1, specificity


if __name__ == '__main__':
    main()