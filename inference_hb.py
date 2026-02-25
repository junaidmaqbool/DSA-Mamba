#!/usr/bin/env python3
"""
Inference script for HB regression models.

Usage:
    python inference_hb.py \
        --model-path checkpoints_transfer/best_model.pth \
        --image-path test_image.jpg \
        --device cuda
        
Or process a batch of images:
    python inference_hb.py \
        --model-path checkpoints_transfer/best_model.pth \
        --image-dir test_images/ \
        --device cuda
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import json

from model.DSAmamba import VSSM as dsamamba


def load_model(model_path, device, num_classes=1):
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    # Create model
    model = dsamamba(in_chans=3, num_classes=num_classes)
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        scaler = checkpoint.get('hb_scaler', None)
    else:
        model.load_state_dict(checkpoint)
        scaler = None
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, scaler


def get_transforms():
    """Get evaluation transforms (ImageNet normalized)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def predict_single_image(model, image_path, device, transform, scaler=None):
    """Predict HB value for a single image."""
    try:
        # Load and transform image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Model inference
        with torch.no_grad():
            output_norm = model(img_tensor).squeeze().cpu().item()
        
        # Denormalize if scaler provided
        if scaler is not None:
            mean, std = scaler
            output = output_norm * std + mean
        else:
            output = output_norm
        
        return output, True, None
    except Exception as e:
        return None, False, str(e)


def predict_batch(model, image_dir, device, transform, scaler=None, max_images=None):
    """Predict HB values for all images in a directory."""
    results = []
    
    # Get list of image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in os.listdir(image_dir) 
                  if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    print(f"Processing {len(image_files)} images from {image_dir}")
    
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        pred, success, error = predict_single_image(model, img_path, device, transform, scaler)
        
        result = {
            'filename': img_name,
            'path': img_path,
            'success': success,
        }
        
        if success:
            result['predicted_hb'] = float(pred)
        else:
            result['error'] = error
        
        results.append(result)
        
        if success:
            print(f"[{idx+1}/{len(image_files)}] {img_name}: HB = {pred:.2f}")
        else:
            print(f"[{idx+1}/{len(image_files)}] {img_name}: ERROR - {error}")
    
    return results


def main():
    parser = argparse.ArgumentParser('HB Regression Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--image-path', type=str, default=None, help='Single image to predict')
    parser.add_argument('--image-dir', type=str, default=None, help='Directory of images to batch predict')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save results as JSON')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to process from directory')
    
    args = parser.parse_args()
    
    if args.image_path is None and args.image_dir is None:
        parser.error("Please provide either --image-path or --image-dir")
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        sys.exit(1)
    
    # Load model
    device = torch.device(args.device)
    model, scaler = load_model(args.model_path, device)
    transform = get_transforms()
    
    print(f"Using device: {device}")
    if scaler is not None:
        print(f"Using scaler: mean={scaler[0]:.3f}, std={scaler[1]:.3f}")
    else:
        print("Note: No scaler found, predictions are normalized values")
    
    print("\n" + "="*60)
    
    results = []
    
    # Process single image
    if args.image_path is not None:
        if not os.path.exists(args.image_path):
            print(f"Error: Image not found at {args.image_path}")
            sys.exit(1)
        
        print(f"Predicting for: {args.image_path}")
        pred, success, error = predict_single_image(model, args.image_path, device, transform, scaler)
        
        if success:
            print(f"\nPredicted HB Value: {pred:.2f}")
            results.append({
                'filename': os.path.basename(args.image_path),
                'predicted_hb': float(pred),
                'success': True
            })
        else:
            print(f"\nError: {error}")
            results.append({
                'filename': os.path.basename(args.image_path),
                'success': False,
                'error': error
            })
    
    # Process batch
    elif args.image_dir is not None:
        if not os.path.isdir(args.image_dir):
            print(f"Error: Directory not found at {args.image_dir}")
            sys.exit(1)
        
        results = predict_batch(model, args.image_dir, device, transform, scaler, args.max_images)
    
    # Save results
    if args.output_json and results:
        print(f"\nSaving results to {args.output_json}...")
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print("Done!")
    
    # Print summary statistics
    if len(results) > 0:
        successful = [r for r in results if r.get('success', False)]
        if successful:
            hb_values = [r['predicted_hb'] for r in successful]
            print("\n" + "="*60)
            print("SUMMARY STATISTICS")
            print("="*60)
            print(f"Total processed: {len(results)}")
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(results) - len(successful)}")
            print(f"\nPredicted HB Values:")
            print(f"  Mean:   {np.mean(hb_values):.2f}")
            print(f"  Median: {np.median(hb_values):.2f}")
            print(f"  Std:    {np.std(hb_values):.2f}")
            print(f"  Min:    {np.min(hb_values):.2f}")
            print(f"  Max:    {np.max(hb_values):.2f}")
            print("="*60)


if __name__ == '__main__':
    main()
