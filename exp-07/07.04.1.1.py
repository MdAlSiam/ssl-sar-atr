import os
from pathlib import Path
import re
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Tuple
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
import cv2
from enum import Enum
from scipy.ndimage import gaussian_filter
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc
)
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, label_binarize
import bm3d
import sys
from datetime import datetime
import seaborn as sns
import json
from tqdm import tqdm
import traceback
import torch.multiprocessing as mp

# Global configurations
RANDOM_SEED = 42
EXP_ID = datetime.now().strftime('%Y%m%d-%H%M%S')
EXPERIMENT_BASE_DIR = os.path.join('exp-07', f'{EXP_ID}')
RESULTS_DIR = os.path.join(EXPERIMENT_BASE_DIR, "results")
FIGURES_DIR = os.path.join(EXPERIMENT_BASE_DIR, "figures")

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Hyperparameters
hyperparameters = {
    'rf_n_estimators': 100,
    'svm_kernel': 'linear',
    'gb_n_estimators': 100,
    'xgb_n_estimators': 100,
    'random_seed': RANDOM_SEED,
    'batch_size': 16, # 16 INPAPER
    'num_epochs': 60,
    'learning_rate': 0.001, # 0.001 INPAPER
    'validation_ratio': 0.15,  # Added when validation added - 0.15 INPAPER
    'early_stopping_patience': 5  # Added when validation added - NOT INPAPER
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class PreextTask(Enum):
    ORIGINAL = 0
    ROTATE_90 = 1
    ROTATE_180 = 2
    ROTATE_270 = 3
    BLUR = 4
    FLIP_HORIZONTAL = 5
    FLIP_VERTICAL = 6
    DENOISE = 7
    ZOOM_IN = 8

def augment_single_image(image, pretext_task):
    """
    GPU-optimized image augmentation function
    """
    # Convert input to tensor and move to GPU
    if torch.is_tensor(image):
        aug_image = image.to('cuda')
    else:
        aug_image = torch.from_numpy(image).to('cuda')
    
    # Create a copy on GPU
    aug_image = aug_image.clone()
    
    # Store original shape
    original_shape = aug_image.shape
    
    # Remove channel dimension if present
    if len(aug_image.shape) == 3:
        aug_image = aug_image.squeeze()
    
    if pretext_task == PreextTask.ORIGINAL:
        pass
        
    elif pretext_task == PreextTask.ROTATE_90:
        aug_image = torch.rot90(aug_image, k=1, dims=(-2, -1))
        
    elif pretext_task == PreextTask.ROTATE_180:
        aug_image = torch.rot90(aug_image, k=2, dims=(-2, -1))
        
    elif pretext_task == PreextTask.ROTATE_270:
        aug_image = torch.rot90(aug_image, k=3, dims=(-2, -1))
        
    elif pretext_task == PreextTask.BLUR:
        # Create Gaussian kernel on GPU
        kernel_size = 5
        sigma = torch.tensor(random.uniform(0.5, 1.0)).to('cuda')
        channels = 1
        
        # Create meshgrid
        x = torch.linspace(-2, 2, kernel_size).to('cuda')
        y = torch.linspace(-2, 2, kernel_size).to('cuda')
        y_grid, x_grid = torch.meshgrid(y, x)
        
        # Create Gaussian kernel
        kernel = torch.exp(-(x_grid.pow(2) + y_grid.pow(2)) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        
        # Reshape kernel for convolution
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        
        # Add batch and channel dimensions to image
        aug_image = aug_image.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        aug_image = F.conv2d(aug_image, kernel, padding=kernel_size//2)
        aug_image = aug_image.squeeze()
        
    elif pretext_task == PreextTask.FLIP_HORIZONTAL:
        aug_image = torch.flip(aug_image, dims=[-1])
        
    elif pretext_task == PreextTask.FLIP_VERTICAL:
        aug_image = torch.flip(aug_image, dims=[-2])
        
    elif pretext_task == PreextTask.DENOISE:
        # For denoising, we'll use a simple Gaussian blur as BM3D is not GPU-compatible
        kernel_size = 3
        sigma = torch.tensor(0.5).to('cuda')
        
        x = torch.linspace(-1, 1, kernel_size).to('cuda')
        y = torch.linspace(-1, 1, kernel_size).to('cuda')
        y_grid, x_grid = torch.meshgrid(y, x)
        
        kernel = torch.exp(-(x_grid.pow(2) + y_grid.pow(2)) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        
        aug_image = aug_image.unsqueeze(0).unsqueeze(0)
        aug_image = F.conv2d(aug_image, kernel, padding=kernel_size//2)
        aug_image = aug_image.squeeze()
    
    elif pretext_task == PreextTask.ZOOM_IN:
        scale_factor = random.uniform(1.2, 1.5)
        
        # Add batch and channel dimensions
        aug_image = aug_image.unsqueeze(0).unsqueeze(0)
        
        # Calculate new dimensions
        h, w = aug_image.shape[-2:]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        # Resize using interpolate
        aug_image = F.interpolate(aug_image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Calculate crop area
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        aug_image = aug_image[..., start_y:start_y+h, start_x:start_x+w]
        aug_image = aug_image.squeeze()
    
    elif pretext_task == PreextTask.ZOOM_OUT:
        scale_factor = random.uniform(0.6, 0.8)
        
        # Add batch and channel dimensions
        aug_image = aug_image.unsqueeze(0).unsqueeze(0)
        
        # Calculate new dimensions
        h, w = aug_image.shape[-2:]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        # Resize using interpolate
        aug_image = F.interpolate(aug_image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Create padding
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        aug_image = F.pad(aug_image, (pad_x, w - new_w - pad_x, pad_y, h - new_h - pad_y), mode='replicate')
        aug_image = aug_image.squeeze()
    
    # Restore channel dimension if it was present in input
    if len(original_shape) == 3:
        aug_image = aug_image.unsqueeze(-1)

    aug_image = aug_image.cpu()
    torch.cuda.empty_cache()
    
    return aug_image, pretext_task.value

class SAMPLEDataset(Dataset):
    def __init__(self, real_root: str, synthetic_root: str, elevation: int = None, transform=None):
        self.real_root = Path(real_root)
        self.synthetic_root = Path(synthetic_root)
        self.transform = transform or transforms.Compose([
            # transforms.Grayscale(),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.class_to_idx = {
            '2s1': 0, 'bmp2': 1, 'btr70': 2, 'm1': 3, 'm2': 4,
            'm35': 5, 'm548': 6, 'm60': 7, 't72': 8, 'zsu23': 9
        }

        self.data = self._load_dataset(elevation)

    def _parse_filename(self, filename: str) -> Dict:
        pattern = r'(\w+)_(real|synthetic|synth)_\w_elevDeg_(\d+)_azCenter_(\d+)_\d+_serial_\w+'
        match = re.match(pattern, filename)
        if match:
            vehicle, data_type, elev, azimuth = match.groups()
            return {
                'vehicle': vehicle,
                'data_type': data_type,
                'elevation': int(elev),
                'azimuth': int(azimuth)
            }
        return None

    def _load_dataset(self, elevation: int = None) -> List[Dict]:
        data = []

        def process_directory(root_path, is_real):
            for class_name in sorted(self.class_to_idx.keys()):  # Sort class names
                class_path = root_path / class_name
                if not class_path.exists():
                    continue

                # Get all image paths and sort them
                image_paths = sorted(list(class_path.glob('*.png')))
                
                for img_path in image_paths:
                    metadata = self._parse_filename(img_path.name)
                    
                    if metadata is None:
                        continue

                    if elevation is not None and metadata['elevation'] != elevation:
                        continue

                    data.append({
                        'path': str(img_path),
                        'label': self.class_to_idx[class_name],
                        'is_real': is_real,
                        'elevation': metadata['elevation'],
                        'azimuth': metadata['azimuth'],
                        'filename': img_path.name
                    })

        process_directory(self.real_root, True)
        process_directory(self.synthetic_root, False)
        
        # Sort the entire dataset by class, then azimuth
        data.sort(key=lambda x: (x['label'], x['azimuth']))
        
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.data[idx]
        image = Image.open(item['path'])
        image = self.transform(image)
        return image, item['label']

class SelfSupervisedDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['path'])
        if self.transform:
            image = self.transform(image)

        # Convert to numpy for augmentation
        image_np = image.numpy().squeeze()
        augmented_images = []
        pretext_labels = []

        for task in PreextTask:
            aug_image, task_label = augment_single_image(image_np, task)
            # aug_tensor = aug_image.cpu()
            aug_tensor = torch.FloatTensor(aug_image).unsqueeze(0)
            augmented_images.append(aug_tensor)
            pretext_labels.append(task_label)

        # Stack tensors (all on CPU)
        stacked_images = torch.stack(augmented_images)
        label_tensor = torch.tensor(pretext_labels)
        
        return stacked_images, label_tensor, item['label']

class DownstreamDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['path'])
        if self.transform:
            image = self.transform(image)
        return image, item['label']

class SARPretrainCNN(nn.Module):
    def __init__(self):
        super(SARPretrainCNN, self).__init__()
        # Feature extractor (same as original SARCNN)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Pretext task classifier
        self.pretext_classifier = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 250),
            nn.ReLU(inplace=True),
            nn.Linear(250, len(PreextTask))
        )
        
    def forward(self, x, return_features=False):
        x = self.features(x)
        features = x.view(x.size(0), -1)  # Flatten

        if return_features:
            return features

        pretext_out = self.pretext_classifier(features)
        return pretext_out

def setup_directories():
    """Create necessary directories for the experiment"""
    try:
        os.makedirs(EXPERIMENT_BASE_DIR, exist_ok=True)
        subdirs = {
            'RESULTS_DIR': RESULTS_DIR,
            'FIGURES_DIR': FIGURES_DIR
        }
        for path in subdirs.values():
            os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        return False
 
def create_data_split_k(dataset, test_elevation=17, k=1.0):
    """Create train/test split based on k-value per paper equations"""
    # Get test data
    test_data = [d for d in dataset.data 
                 if d['elevation'] == test_elevation and d['is_real']]

    # Print test distribution
    test_classes = [d['label'] for d in test_data]
    print("\nTest data class distribution:")
    for label, count in Counter(test_classes).items():
        print(f"Class {label}: {count} samples")
    
    # Get training data
    class_data = defaultdict(list)
    for d in dataset.data:
        if d['elevation'] != test_elevation and d['is_real']:
            class_data[d['label']].append(d)

    # Print available training data
    # print("\nAvailable training data before k-selection:")
    # for label, data in class_data.items():
    #     print(f"Class {label}: {len(data)} samples")
                
    print("\nTraining data selection with k={:.2f}:".format(k))
    training_data = []
    for class_label in class_data.keys():
        available_samples = len(class_data[class_label])
        # Simply take k portion of available samples
        samples_to_take = int(np.ceil(k * available_samples))
        
        print(f"Class {class_label}:", end=' ')
        print(f"Available samples: {available_samples}", end=' | ')
        print(f"Taking {samples_to_take} samples (k={k:.2f})")
        
        # Take first k portion of the data
        training_data.extend(class_data[class_label][:samples_to_take])

    # Add to create_data_split_k
    with open(f"{RESULTS_DIR}/k_{k:.2f}_sample_selection.txt", 'w') as f:
        f.write(f"Samples selected for k={k:.2f}:\n")
        for sample in training_data:
            f.write(f"Class: {sample['label']}, Azimuth: {sample['azimuth']}, File: {sample['filename']}\n")
    
    return training_data, test_data

def create_train_val_split(train_data, val_ratio=0.15):
    """Split training data into train and validation sets"""
    total_samples = len(train_data)
    val_size = int(total_samples * val_ratio)
    
    # Create a random permutation of indices
    indices = torch.randperm(total_samples).tolist()
    
    # Split indices
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Split the data
    train_subset = [train_data[i] for i in train_indices]
    val_subset = [train_data[i] for i in val_indices]
    
    return train_subset, val_subset

def train_pretext(model, train_loader, optimizer, criterion, device):
    """GPU-optimized pretext training function with mixed precision"""
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    # Move criterion to GPU
    criterion = criterion.to(device)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    
    for batch_imgs, batch_pretext_labels, _ in train_loader:
        # Pre-allocate tensors on GPU
        batch_imgs = batch_imgs.view(-1, 1, 64, 64).to(device, non_blocking=True)
        batch_pretext_labels = batch_pretext_labels.view(-1).to(device, non_blocking=True)
        
        # Zero grad with set_to_none for better performance
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        with autocast():
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_pretext_labels)
        
        # Scale and backpropagate loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        # Optional: Clear cache periodically if memory is an issue
        if batch_count % 100 == 0:
            torch.cuda.empty_cache()
    
    return epoch_loss / batch_count

def validate_pretext(model, val_loader, criterion, device):
    """Validate pretext task performance"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_imgs, batch_pretext_labels, _ in val_loader:
            batch_imgs = batch_imgs.view(-1, 1, 64, 64).to(device, non_blocking=True)
            batch_pretext_labels = batch_pretext_labels.view(-1).to(device, non_blocking=True)
            
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_pretext_labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += batch_pretext_labels.size(0)
            correct += predicted.eq(batch_pretext_labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def extract_features(model, dataloader, device):
    """GPU-optimized feature extraction"""
    model.eval()
    features = []
    labels = []
    
    # Pre-allocate GPU memory for batches
    with torch.no_grad(), torch.cuda.amp.autocast():  # Enable automatic mixed precision
        for images, class_labels in dataloader:
            images = images.to(device, non_blocking=True)
            batch_features = model(images, return_features=True)
            # Keep features on GPU until batch is complete
            features.append(batch_features)
            labels.extend(class_labels.numpy())
    
    # Concatenate all features on GPU first, then move to CPU
    features = torch.cat(features, dim=0)
    features_np = features.cpu().numpy()
    
    return features_np, np.array(labels)

def train_downstream_classifiers(train_features, train_labels, test_features, test_labels):
    """Train and evaluate downstream classifiers"""
    results = {}
    trained_classifiers = {}
    
    # First move everything to CPU and convert to numpy
    if torch.is_tensor(train_features):
        train_features = train_features.cpu().numpy()
    if torch.is_tensor(test_features):
        test_features = test_features.cpu().numpy()
    if torch.is_tensor(train_labels):
        train_labels = train_labels.cpu().numpy()
    if torch.is_tensor(test_labels):
        test_labels = test_labels.cpu().numpy()
    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    classifiers = {
        'xgboost': xgb.XGBClassifier(
            n_estimators=hyperparameters['xgb_n_estimators'],
            random_state=hyperparameters['random_seed'],
            use_label_encoder=False,
            tree_method='gpu_hist',  # Only use if you have CUDA toolkit installed
            predictor='gpu_predictor',
            gpu_id=0,
            verbose=0,
            objective='multi:softprob',
            num_class=10
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=hyperparameters['rf_n_estimators'],
            random_state=hyperparameters['random_seed'],
            verbose=0,
            n_jobs=-1
        ),
        'svm': SVC(
            kernel=hyperparameters['svm_kernel'],
            random_state=hyperparameters['random_seed'],
            verbose=False,
            probability=True
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=hyperparameters['gb_n_estimators'],
            random_state=hyperparameters['random_seed'],
            verbose=0
        )
    }
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        try:
            if name == 'xgboost':
                try:
                    # Try GPU-enabled XGBoost
                    clf.fit(train_features_scaled, train_labels)
                except (ImportError, Exception) as e:
                    print(f"GPU training failed for XGBoost: {str(e)}")
                    print("Falling back to CPU XGBoost...")
                    # Fall back to CPU XGBoost
                    clf = xgb.XGBClassifier(
                        n_estimators=hyperparameters['xgb_n_estimators'],
                        random_state=hyperparameters['random_seed'],
                        use_label_encoder=False,
                        verbose=0,
                        objective='multi:softprob',
                        num_class=10
                    )
            
            clf.fit(train_features_scaled, train_labels)
            y_pred = clf.predict(test_features_scaled)
            
            metrics = {
                'accuracy': accuracy_score(test_labels, y_pred),
                'precision': precision_score(test_labels, y_pred, average='macro'),
                'recall': recall_score(test_labels, y_pred, average='macro'),
                'f1': f1_score(test_labels, y_pred, average='macro'),
                'confusion_matrix': confusion_matrix(test_labels, y_pred)
            }
            
            results[name] = metrics
            trained_classifiers[name] = clf
            
            print(f"{name} Metrics:")
            for metric_name, value in metrics.items():
                if metric_name != 'confusion_matrix':
                    print(f"{metric_name}: {value:.4f}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            results[name] = None
            trained_classifiers[name] = None
            continue
    
    return results, trained_classifiers, test_features_scaled, test_labels

def evaluate_downstream_task(clf, X_test, y_test, pretext_classifier, k_value, exp_id, run_number, save_dir=FIGURES_DIR):
    """
    Evaluate downstream classifier with ROC curves and metrics
    """
    # Get predictions
    y_pred = clf.predict(X_test)
    clf_name = clf.__class__.__name__.lower()
    
    # Determine if binary or multiclass
    n_classes = len(np.unique(y_test))
    is_binary = n_classes == 2

    # Initialize metrics dictionary
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    # Check if classifier supports probability predictions
    has_proba = hasattr(clf, 'predict_proba')
    has_decision = hasattr(clf, 'decision_function')

    if not (has_proba or has_decision):
        print(f"Warning: {clf_name} doesn't support probability estimates or decision function. Skipping ROC curve.")
        return metrics

    # Define FPR thresholds of interest
    fpr_thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]
    
    plt.figure(figsize=(10, 8))
    
    if is_binary:
        # Binary classification case
        if has_proba:
            y_scores = clf.predict_proba(X_test)[:, 1]
        else:
            y_scores = clf.decision_function(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        
        # Calculate TPR at specific FPR thresholds
        tpr_at_thresholds = {}
        for threshold in fpr_thresholds:
            idx = np.argmin(np.abs(fpr - threshold))
            tpr_at_thresholds[threshold] = tpr[idx]
            plt.plot(fpr[idx], tpr[idx], 'ro', label=f'TPR at {threshold*100}% FPR = {tpr[idx]:.3f}')
        
        # CHANGE 1: Update metrics with string keys for binary case
        metrics.update({
            'roc_auc': roc_auc,
            'tpr_at_thresholds': {str(k): v for k, v in tpr_at_thresholds.items()}
        })

    else:
        # Multiclass case
        if has_proba:
            y_scores = clf.predict_proba(X_test)
        else:
            decision_values = clf.decision_function(X_test)
            if decision_values.ndim > 1:
                y_scores = np.exp(decision_values) / np.sum(np.exp(decision_values), axis=1, keepdims=True)
            else:
                y_scores = np.column_stack([1 - decision_values, decision_values])

        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        
        # Initialize dictionaries
        fpr = dict()
        tpr = dict()

        # Compute ROC curve and ROC area for each class
        class_names = ['2S1', 'BMP2', 'BTR70', 'M1', 'M2', 'M35', 'M548', 'M60', 'T72', 'ZSU23']
        class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        # Calculate ROC for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
            roc_auc_class = auc(fpr[i], tpr[i])
            
            # Plot individual class ROC curves
            plt.plot(
                fpr[i], 
                tpr[i], 
                color=class_colors[i],
                lw=1,  # thinner lines for class curves
                label=f'{class_names[i]} (AUC = {roc_auc_class:.4f})'
            )
        
        # Compute micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test_bin.ravel(), 
            y_scores.ravel()
        )
        roc_auc = auc(fpr["micro"], tpr["micro"])
        
        plt.plot(
            fpr["micro"], 
            tpr["micro"],
            label=f'micro-average ROC curve (AUC = {roc_auc:.4f})',
            color='deeppink', 
            linestyle=':', 
            linewidth=3
        )
        
        # Calculate TPR at specific FPR thresholds for micro-average
        tpr_at_thresholds = {}
        for threshold in fpr_thresholds:
            idx = np.argmin(np.abs(fpr["micro"] - threshold))
            tpr_at_thresholds[threshold] = tpr["micro"][idx]
            plt.plot(fpr["micro"][idx], tpr["micro"][idx], 'ro', 
                    label=f'Micro-avg TPR at {threshold*100}% FPR = {tpr["micro"][idx]:.3f}')
        
        # CHANGE 2: Update metrics with nested structure for multiclass case
        # Calculate per-class AUC values
        class_auc = {}
        for i in range(n_classes):
            class_auc[class_names[i]] = auc(fpr[i], tpr[i])
        
        metrics.update({
            'roc_auc': roc_auc,  # micro-average AUC
            'roc_auc_per_class': class_auc,  # per-class AUC values
            'tpr_at_thresholds': {'micro': {str(k): v for k, v in tpr_at_thresholds.items()}}
        })

    # Complete the ROC plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{exp_id}-k_{k_value:.2f}-run_{run_number}-{pretext_classifier}-{clf_name}-ROC")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            save_dir, 
            f"{exp_id}-k_{k_value:.2f}-run_{run_number}-{pretext_classifier}-{clf_name}-roc.png"
        ),
        bbox_inches='tight'
    )
    plt.close()
    
    return metrics

def plot_confusion_matrices(results, k, save_path):
    """Plot and save confusion matrices"""
    cm = results['confusion_matrix']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (k={k:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrices_2(results, k, save_path):
    """Plot and save confusion matrices with both counts and percentages"""
    cm = results['confusion_matrix']
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    
    class_labels = ['2S1', 'BMP2', 'BTR70', 'M1', 'M2', 'M35', 'M548', 'M60', 'T72', 'ZSU23']
    
    # Create a copy of cm_percent for displaying but replacing 0 with actual 0.00
    display_cm = np.zeros_like(cm_percent)
    annotations = []
    for i in range(cm.shape[0]):
        row_annotations = []
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                row_annotations.append(f'0.00\n0')
                display_cm[i, j] = 0
            else:
                row_annotations.append(f'{cm_percent[i, j]:.2f}\n{cm[i, j]}')
                display_cm[i, j] = cm_percent[i, j]
        annotations.append(row_annotations)
    
    annotations = np.array(annotations)
    
    # Create heatmap with both percentage colors and count annotations
    sns.heatmap(display_cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_labels, 
                yticklabels=class_labels,
                annot_kws={'size': 10})
    
    plt.title(f'Confusion Matrix (k={k:.2f}, Accuracy: {results["accuracy"]:.2%})')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    # Adjust layout
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_k_results(k_values, results, detailed_metrics, save_path):
    """Plot k vs accuracy results with error bars from multiple runs"""
    plt.figure(figsize=(12, 8))
    
    for clf_name, accuracies_dict in results.items():
        # Convert the dictionary values to a proper numpy array
        accuracies_list = []
        for k in k_values:
            if k in accuracies_dict:
                accuracies_list.append(accuracies_dict[k])
        
        # Convert to numpy array ensuring consistent shape
        accuracies = np.array(accuracies_list)
        
        # Calculate mean and std
        mean_acc = np.mean(accuracies, axis=1)
        std_acc = np.std(accuracies, axis=1)
        
        plt.plot(k_values, mean_acc, label=f'{clf_name} (mean)', marker='o')
        plt.fill_between(k_values, 
                        mean_acc - std_acc, 
                        mean_acc + std_acc, 
                        alpha=0.2)
    
    plt.xlabel('Fraction of Measured Training Data (k)')
    plt.ylabel('Classification Accuracy')
    plt.title('Self-Supervised Learning Results (with std dev)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_detailed_metrics(metrics, k_values, filepath):
    """Save detailed metrics with reduced TPR/FPR data"""
    output = {
        'k_values': k_values.tolist(),
        'metrics': {}
    }

    for k, v in metrics.items():
        output['metrics'][str(k)] = {
            'per_run': {},
            'mean': {},
            'std': {}
        }
        
        for metric_name, values in v.items():
            # CHANGE 3: Skip storing raw TPR/FPR arrays to reduce JSON size
            if metric_name.endswith('_fpr') or metric_name.endswith('_tpr'):
                continue
                
            # Store other metrics
            output['metrics'][str(k)]['per_run'][metric_name] = values
            
            if isinstance(values[0], (int, float, np.number)):
                output['metrics'][str(k)]['mean'][metric_name] = float(np.mean(values))
                output['metrics'][str(k)]['std'][metric_name] = float(np.std(values))
            else:
                output['metrics'][str(k)]['mean'][metric_name] = values[0]
                output['metrics'][str(k)]['std'][metric_name] = 0

    # Save to file
    with open(filepath, 'w') as f:
        json.dump(output, f, cls=NumpyEncoder, indent=2)

def run_self_supervised_experiment(real_root, synthetic_root, test_elevation=17, n_runs=10):
    """Main experiment function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[DEVICE] {device}')

    k_values = np.arange(0.05, 1.05, 0.05) # 0.00, 1.05, 0.05 INPAPER
    results = defaultdict(lambda: defaultdict(list))
    detailed_metrics = defaultdict(lambda: defaultdict(list))

    dataset = SAMPLEDataset(real_root, synthetic_root)

    total_steps = len(k_values) * n_runs
    step = 0
    
    for k_idx, k in enumerate(k_values):
        print(f"\n{'='*125}")
        print(f"Processing k={k:.2f} ({k_idx+1}/{len(k_values)})")
        print(f"{'='*125}")
        
        k_results = defaultdict(list)
        
        for run in range(n_runs):
            step += 1
            print(f"\n-------------------------------------- Run {run+1}/{n_runs} (Overall progress: {step}/{total_steps})  --------------------------------------------------------- ")
            
            # Create k-based split
            train_data, test_data = create_data_split_k(dataset, test_elevation, k)
            print(f"\nTotal training samples before split: {len(train_data)}")

            # Split training data into train and validation
            train_subset, val_subset = create_train_val_split(train_data, val_ratio=hyperparameters['validation_ratio'])
            print(f"Training samples: {len(train_subset)}")
            print(f"Validation samples: {len(val_subset)}")

            # Create datasets:
            # 1. For pretext task training (with augmentations)
            pretext_train_dataset = SelfSupervisedDataset(train_subset, dataset.transform)
            pretext_val_dataset = SelfSupervisedDataset(val_subset, dataset.transform)

            pretext_train_loader = DataLoader(
                pretext_train_dataset, 
                batch_size=hyperparameters['batch_size'], 
                shuffle=True, 
                pin_memory=True,
                num_workers=4,  
                prefetch_factor=2
            )

            pretext_val_loader = DataLoader(
                pretext_val_dataset,
                batch_size=hyperparameters['batch_size'],
                shuffle=False,  # No need to shuffle validation data
                pin_memory=True,
                num_workers=4,
                prefetch_factor=2
            )

            # 2. For feature extraction (without augmentations)
            downstream_train_dataset = DownstreamDataset(train_subset, dataset.transform)  # Note: using train_subset
            downstream_test_dataset = DownstreamDataset(test_data, dataset.transform)
            
            downstream_train_loader = DataLoader(
                downstream_train_dataset, 
                batch_size=hyperparameters['batch_size'],
                shuffle=True,
                pin_memory=True,
                num_workers=4,  
                prefetch_factor=2
            )
            downstream_test_loader = DataLoader(
                downstream_test_dataset, 
                batch_size=hyperparameters['batch_size'], 
                shuffle=False,
                num_workers=4,  
                prefetch_factor=2
            )
            
            # Train pretext model
            pretext_model = SARPretrainCNN().to(device)
            optimizer = optim.Adam(pretext_model.parameters(), lr=hyperparameters['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            best_val_loss = float('inf')
            patience = hyperparameters['early_stopping_patience']
            patience_counter = 0
            best_model_state = None

            print("\nStarting pretext training:")
            for epoch in range(hyperparameters['num_epochs']):
                # Training
                train_loss = train_pretext(pretext_model, pretext_train_loader, optimizer, criterion, device)
                
                # Validation
                val_loss, val_acc = validate_pretext(pretext_model, pretext_val_loader, criterion, device)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(pretext_model.state_dict())
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            # Load best model for feature extraction if early stopping occurred
            if best_model_state is not None:
                pretext_model.load_state_dict(best_model_state)
                
            # Extract features using original images only
            train_features, train_labels = extract_features(
                pretext_model, 
                downstream_train_loader,  # Using non-augmented loader
                device
            )
            test_features, test_labels = extract_features(
                pretext_model, 
                downstream_test_loader,   # Using non-augmented loader
                device
            )
            
            # Train and evaluate downstream classifiers
            clf_results, trained_clfs, test_features_scaled, test_labels = train_downstream_classifiers(
                train_features, train_labels,
                test_features, test_labels
            )
            
            # Generate ROC curves and additional metrics
            for clf_name, clf in trained_clfs.items():
                if clf is not None:
                    metrics = evaluate_downstream_task(
                        clf,
                        test_features_scaled,
                        test_labels,
                        pretext_classifier='cnn',
                        k_value=k,
                        exp_id=EXP_ID,
                        run_number=run,
                        save_dir=FIGURES_DIR
                    )
                    
                    for metric_name, value in metrics.items():
                        if metric_name != 'confusion_matrix':
                            detailed_metrics[k][f"{clf_name}_{metric_name}"].append(value)
                    
                    k_results[clf_name].append(metrics['accuracy'])
            
            # Save confusion matrices for k=0 and k=1 cases
            # if k in [0.05, 1.0]:
            for clf_name, clf_results in clf_results.items():
                if clf_results is not None:
                    plot_confusion_matrices(
                        clf_results,
                        k,
                        f"{FIGURES_DIR}/{EXP_ID}-conf_matrix_k_{k:.2f}_run_{run}_{clf_name}.png"
                    )
                    plot_confusion_matrices_2(
                        clf_results,
                        k,
                        f"{FIGURES_DIR}/{EXP_ID}-conf_matrix_k_{k:.2f}_run_{run}_{clf_name}-2.png"
                    )
        
        # After all runs for this k value, store in main results dictionary
        for clf_name, accuracies in k_results.items():
            results[clf_name][k] = accuracies  # Store k_results in main results
            
        # Print average results for this k
        print(f"\nResults for k={k:.2f}:")
        for clf_name, accuracies in k_results.items():
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"{clf_name}:")
            print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            
            # Print ROC metrics if they exist
            roc_key = f"{clf_name}_roc_auc"
            if roc_key in detailed_metrics[k]:
                auc_values = detailed_metrics[k][roc_key]
                mean_auc = np.mean(auc_values)
                std_auc = np.std(auc_values)
                print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
                
                # CHANGE 4: Updated TPR calculation to handle both binary and multiclass cases
                tpr_key = f"{clf_name}_tpr_at_thresholds"
                if tpr_key in detailed_metrics[k]:
                    print("  Mean Micro-average TPR at FPR thresholds:")
                    tpr_values = detailed_metrics[k][tpr_key]
                    for fpr_threshold in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]:
                        # Extract TPR values for this threshold across all runs
                        threshold_tprs = []
                        for run_tprs in tpr_values:
                            if isinstance(run_tprs, dict):
                                # If multiclass, get micro average TPRs
                                if 'micro' in run_tprs:
                                    micro_tprs = run_tprs.get('micro', {})
                                    closest_threshold = min(
                                        [float(t) for t in micro_tprs.keys()], 
                                        key=lambda x: abs(x - fpr_threshold)
                                    )
                                    threshold_tprs.append(micro_tprs[str(closest_threshold)])
                                else:
                                    # For binary classification
                                    closest_threshold = min(
                                        [float(t) for t in run_tprs.keys()], 
                                        key=lambda x: abs(x - fpr_threshold)
                                    )
                                    threshold_tprs.append(run_tprs[str(closest_threshold)])
                        
                        if threshold_tprs:
                            mean_tpr = np.mean(threshold_tprs)
                            std_tpr = np.std(threshold_tprs)
                            print(f"    FPR {fpr_threshold:.2f}: TPR = {mean_tpr:.4f} ± {std_tpr:.4f}")

    # Convert results to format needed for plotting
    final_results = {clf_name: [values[k] for k in k_values] 
                    for clf_name, values in results.items()}

    # Before plotting, prepare results correctly
    final_results = {}
    for clf_name, values in results.items():
        accuracies_by_k = []
        for k in k_values:
            if k in values:
                accuracies_by_k.append(values[k])
            else:
                accuracies_by_k.append([])
        final_results[clf_name] = accuracies_by_k
    
    # Plot results with error bars
    plot_k_results(k_values, final_results, detailed_metrics, f"{FIGURES_DIR}/{EXP_ID}-k_accuracy.png")
    
    # Save detailed metrics
    save_detailed_metrics(detailed_metrics, k_values, f"{RESULTS_DIR}/{EXP_ID}-detailed_metrics-411.json")
    
    return k_values, results, detailed_metrics

class MultiWriter:
    def __init__(self, filename):
        self.log = open(filename, 'w', encoding='utf-8')
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Ensure directories exist
    if not setup_directories():
        print("Failed to create necessary directories. Exiting.")
        sys.exit(1)

    # Setup logging
    timestamp = datetime.now().strftime('%Y-%m-%d-%Hh-%Mm-%Ss')
    filename = os.path.join(RESULTS_DIR, f"{EXP_ID}-log-411.txt")
    print(f"> Output will be saved to: {filename}")

    try:
        # Setup logging
        # original_stdout = sys.stdout
        sys.stdout = MultiWriter(filename)

        # Define paths
        real_root = '/home/arni-linux/pytorch_project/SAMPLE_dataset_public/png_images/qpm/real'
        synthetic_root = '/home/arni-linux/pytorch_project/SAMPLE_dataset_public/png_images/qpm/synth'

        # Run experiment and time it
        print(f"=== Starting Experiment {EXP_ID} ===")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run self supervised experiment
        start_time = datetime.now()
        k_values, results, detailed_metrics = run_self_supervised_experiment(real_root, synthetic_root)
        end_time = datetime.now()
        
        # Calculate and print duration
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n=== Experiment Complete ===")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        # Save final results
        final_results = {
            'k_values': k_values.tolist(),
            'classifier_results': results,
            'detailed_metrics': detailed_metrics,
            'experiment_info': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'hyperparameters': hyperparameters
            }
        }
        
        with open(os.path.join(RESULTS_DIR, f"{EXP_ID}_final_results-411.json"), 'w') as f:
            json.dump(final_results, f, cls=NumpyEncoder, indent=2)
        
        print(f"\n=== Results Saved Successfully of Experiment {EXP_ID} ===")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(traceback.print_exc())
    finally:
        # Restore stdout and close log file
        if isinstance(sys.stdout, MultiWriter):
            sys.stdout.log.close()
        sys.stdout = sys.__stdout__
        print(f"=== PROCESS FINISHED: {EXP_ID} ===")