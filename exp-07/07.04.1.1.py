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
from typing import Dict, List, Tuple
import pandas as pd
from collections import Counter, defaultdict
from tabulate import tabulate
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
import cv2
from enum import Enum
from scipy.ndimage import gaussian_filter
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import bm3d
import sys
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

RANDOM_SEED = 42
EXPERIMENT_BASE_DIR = 'exp-07'
EXP_ID = datetime.now().strftime('%Y%m%d-%H%M%S')
MODEL_DIR = os.path.join(EXPERIMENT_BASE_DIR, "pretext_models")
DATA_DIR = os.path.join(EXPERIMENT_BASE_DIR, "processed_data")
RESULTS_DIR = os.path.join(EXPERIMENT_BASE_DIR, "results")
FIGURES_DIR = os.path.join(EXPERIMENT_BASE_DIR, "figures")

class SAMPLEDataset(Dataset):
    def __init__(self, real_root: str, synthetic_root: str, elevation: int = None, transform=None):
        self.real_root = Path(real_root)
        self.synthetic_root = Path(synthetic_root)
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
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
            for class_name in self.class_to_idx.keys():
                class_path = root_path / class_name
                if not class_path.exists():
                    continue

                for img_path in class_path.glob('*.png'):
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
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.data[idx]
        image = Image.open(item['path'])
        image = self.transform(image)
        return image, item['label']

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

def zoom_image(image, scale_factor):
    """
    Apply zoom to image with padding or cropping
    
    Parameters:
        image: Input image array
        scale_factor: > 1 for zoom-in, < 1 for zoom-out
    """
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    if scale_factor > 1:  # Zoom in
        # Resize image to larger size
        zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Calculate crop area to maintain original size
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        zoomed = zoomed[start_y:start_y+h, start_x:start_x+w]

    else:  # Zoom out
        # Resize image to smaller size
        zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create padding
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2

        zoomed = cv2.copyMakeBorder(
            zoomed,
            top=pad_y,
            bottom=h - new_h - pad_y,
            left=pad_x,
            right=w - new_w - pad_x,
            borderType=cv2.BORDER_REPLICATE
        )

    return zoomed

def augment_single_image(image, pretext_task):
    """
    Apply specific pretext task transformation to image
    """
    aug_image = image.copy()
    original_shape = aug_image.shape
    
    # Remove channel dimension for operations that expect 2D input
    if len(aug_image.shape) == 3:
        aug_image = aug_image.squeeze()
    
    if pretext_task == PreextTask.ORIGINAL:
        pass
        
    elif pretext_task == PreextTask.ROTATE_90:
        rows, cols = aug_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        aug_image = cv2.warpAffine(aug_image, M, (cols, rows))
        
    elif pretext_task == PreextTask.ROTATE_180:
        rows, cols = aug_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
        aug_image = cv2.warpAffine(aug_image, M, (cols, rows))
        
    elif pretext_task == PreextTask.ROTATE_270:
        rows, cols = aug_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
        aug_image = cv2.warpAffine(aug_image, M, (cols, rows))
        
    elif pretext_task == PreextTask.BLUR:
        sigma = random.uniform(0.5, 1.0)
        aug_image = gaussian_filter(aug_image, sigma=sigma)
        
    elif pretext_task == PreextTask.FLIP_HORIZONTAL:
        aug_image = cv2.flip(aug_image, 1)
        
    elif pretext_task == PreextTask.FLIP_VERTICAL:
        aug_image = cv2.flip(aug_image, 0)
        
    elif pretext_task == PreextTask.DENOISE:
        sigma_psd = 25/255
        aug_image = bm3d.bm3d(aug_image, sigma_psd=sigma_psd)
    
    elif pretext_task == PreextTask.ZOOM_IN:
        scale_factor = random.uniform(1.2, 1.5)  # Zoom in 20-50%
        aug_image = zoom_image(aug_image, scale_factor)
        
    elif pretext_task == PreextTask.ZOOM_OUT:
        scale_factor = random.uniform(0.6, 0.8)  # Zoom out 20-40%
        aug_image = zoom_image(aug_image, scale_factor)

    # Restore channel dimension if it was present in input
    if len(original_shape) == 3:
        aug_image = aug_image[..., np.newaxis]
        
    return aug_image, pretext_task.value

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
            aug_tensor = torch.FloatTensor(aug_image).unsqueeze(0)
            augmented_images.append(aug_tensor)
            pretext_labels.append(task_label)

        return torch.stack(augmented_images), torch.tensor(pretext_labels), item['label']

class SARPretrainCNN(nn.Module):
    def __init__(self):
        super(SARPretrainCNN, self).__init__()
        # Feature extractor (same as original SARCNN)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Pretext task classifier
        self.pretext_classifier = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, len(PreextTask))
        )
        
    def forward(self, x, return_features=False):
        x = self.features(x)
        features = x.view(x.size(0), -1)  # Flatten

        if return_features:
            return features

        pretext_out = self.pretext_classifier(features)
        return pretext_out

def create_data_split(dataset, test_elevation=17, k=1.0):
    """Modified to only use measured data without synthetic replacement"""
    test_data = [d for d in dataset.data 
                 if d['elevation'] == test_elevation and d['is_real']]

    # Group remaining data by class
    class_data = defaultdict(list)
    for d in dataset.data:
        if d['elevation'] != test_elevation and d['is_real']:  # Only measured data
            class_data[d['label']].append(d)

    # Create training data
    training_data = []
    for class_label, data in class_data.items():
        Nm_j = len([d for d in dataset.data 
                   if d['label'] == class_label and d['is_real']])
        Sm_j = len([d for d in test_data if d['label'] == class_label])
        
        # Calculate number of training samples
        Tm_j = int(k * (Nm_j - Sm_j))

        training_data.extend(data[:Tm_j])

    return training_data, test_data

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch_imgs, _, batch_labels in dataloader:
            # Get original images only (first augmentation)
            orig_imgs = batch_imgs[:, 0].to(device)
            batch_features = model(orig_imgs, return_features=True)
            features.append(batch_features.cpu().numpy())
            labels.extend(batch_labels.numpy())
            
    return np.vstack(features), np.array(labels)

# Define global hyperparameters
hyperparameters = {
    'rf_n_estimators': 100,
    'svm_kernel': 'linear',
    'gb_n_estimators': 100,
    'xgb_n_estimators': 100,
    'cv_splits': 5,
    'random_seed': 42
}

def train_downstream_classifiers(train_features, train_labels, test_features, test_labels):
    """
    Train and evaluate multiple classifiers with specified hyperparameters.
    
    Args:
        train_features: Training feature matrix
        train_labels: Training labels
        test_features: Test feature matrix
        test_labels: Test labels
        
    Returns:
        results: Dictionary containing evaluation metrics for each classifier
        trained_classifiers: Dictionary containing trained classifier objects
    """
    results = {}
    trained_classifiers = {}
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Initialize classifiers with hyperparameters
    classifiers = {
        'random_forest': RandomForestClassifier(
            n_estimators=hyperparameters['rf_n_estimators'],
            random_state=hyperparameters['random_seed'],
            verbose=0,
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
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=hyperparameters['xgb_n_estimators'],
            random_state=hyperparameters['random_seed'],
            use_label_encoder=False,  # For newer XGBoost versions
            verbose=0
        )
    }
    
    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        try:
            # Train the classifier
            clf.fit(train_features_scaled, train_labels)
            
            # Make predictions
            y_pred = clf.predict(test_features_scaled)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(test_labels, y_pred),
                'precision': precision_score(test_labels, y_pred, average='macro'),
                'recall': recall_score(test_labels, y_pred, average='macro'),
                'f1': f1_score(test_labels, y_pred, average='macro'),
                'confusion_matrix': confusion_matrix(test_labels, y_pred)
            }
            
            # Store results and trained classifier
            results[name] = metrics
            trained_classifiers[name] = clf
            
            print(f"{name} Accuracy: {metrics['accuracy']:.2f}")
            print(f"{name} Precision: {metrics['precision']:.2f}")
            print(f"{name} Recall: {metrics['recall']:.2f}")
            print(f"{name} F1: {metrics['f1']:.2f}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            results[name] = None
            trained_classifiers[name] = None
            continue
    
    return results, trained_classifiers, test_features_scaled, test_labels

def run_self_supervised_experiment(real_root, synthetic_root, test_elevation=17, n_runs=5):
    """
    Run self-supervised learning experiment with multiple classifiers and evaluations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_values = np.arange(0.05, 1.05, 0.05)
    results = defaultdict(list)
    
    # Store detailed metrics for each k value
    detailed_metrics = defaultdict(lambda: defaultdict(list))
    
    dataset = SAMPLEDataset(real_root, synthetic_root)
    
    print(f'DEVICE STATUS: {device}')
    total_steps = len(k_values) * n_runs
    step = 0
    
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Processing k={k:.2f} ({k_values.tolist().index(k)+1}/{len(k_values)})")
        print(f"{'='*50}")
        
        k_results = defaultdict(list)
        
        for run in range(n_runs):
            step += 1
            print(f"\nRun {run+1}/{n_runs} (Overall progress: {step}/{total_steps})")
            
            # Create k-based split
            train_data, test_data = create_data_split_k(dataset, test_elevation, k)
            print(f"Training with {len(train_data)} samples, Testing with {len(test_data)} samples")
            
            # Create self-supervised datasets
            train_dataset = SelfSupervisedDataset(train_data, dataset.transform)
            test_dataset = SelfSupervisedDataset(test_data, dataset.transform)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            # Train pretext model
            pretext_model = SARPretrainCNN().to(device)
            optimizer = optim.Adam(pretext_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Pretext task training
            total_epochs = 60
            for epoch in range(total_epochs):
                model.train()
                epoch_loss = 0
                batch_count = 0

                if epoch % 10 == 0:  # Print every 10 epochs
                    print(f"Epoch {epoch}/{total_epochs}")

                for batch_idx, (batch_imgs, batch_pretext_labels, _) in enumerate(train_loader):
                    batch_imgs = batch_imgs.view(-1, 1, 64, 64).to(device)
                    batch_pretext_labels = batch_pretext_labels.view(-1).to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_imgs)
                    loss = criterion(outputs, batch_pretext_labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1

                if epoch % 10 == 0:  # Print loss every 10 epochs
                    avg_loss = epoch_loss / batch_count
                    print(f"Average loss: {avg_loss:.4f}")
                
            # Extract features for downstream task
            train_features, train_labels = extract_features(pretext_model, train_loader, device)
            test_features, test_labels = extract_features(pretext_model, test_loader, device)
            
            # Train and evaluate downstream classifiers with new function
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
                        save_dir=FIGURES_DIR
                    )
                    
                    # Store metrics
                    for metric_name, value in metrics.items():
                        if metric_name != 'confusion_matrix':  # Don't store large arrays
                            detailed_metrics[k][f"{clf_name}_{metric_name}"].append(value)
                    
                    # Store accuracy for main results
                    k_results[clf_name].append(metrics['accuracy'])
            
            # Save confusion matrices for k=0 and k=1 cases
            if k in [0.0, 1.0]:
                for clf_name, results in clf_results.items():
                    if results is not None:
                        plot_confusion_matrices(
                            results,
                            k,
                            f"{FIGURES_DIR}/conf_matrix_k{k}_{clf_name}.png"
                        )
        
        # Average results for this k
        for clf_name in k_results.keys():
            avg_accuracy = np.mean(k_results[clf_name])
            results[clf_name].append(avg_accuracy)
            print(f"\nk={k:.2f}, {clf_name} Average Accuracy: {avg_accuracy:.4f}")
    
    # Plot results similar to Figure 4
    plot_k_results(k_values, results, f"{FIGURES_DIR}/k_accuracy.png")
    
    # Save detailed metrics
    save_detailed_metrics(detailed_metrics, k_values, f"{RESULTS_DIR}/detailed_metrics.json")
    
    return k_values, results, detailed_metrics

def save_detailed_metrics(metrics, k_values, filepath):
    """Save detailed metrics to JSON file"""
    output = {
        'k_values': k_values.tolist(),
        'metrics': {str(k): v for k, v in metrics.items()}
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, cls=NumpyEncoder)

def plot_k_results(k_values, results, save_path):
    plt.figure(figsize=(12, 8))
    for clf_name, accuracies in results.items():
        plt.plot(k_values, accuracies, label=clf_name, marker='o')
    plt.xlabel('Fraction of Measured Training Data (k)')
    plt.ylabel('Classification Accuracy (%)')
    plt.title('Self-Supervised Learning Results')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def evaluate_downstream_task(clf, X_test, y_test, pretext_classifier, k_value, exp_id, save_dir=FIGURES_DIR):
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
    fpr_thresholds = [0.01, 0.05, 0.10, 0.20, 0.25]
    
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
        
        metrics.update({
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'tpr_at_thresholds': tpr_at_thresholds
        })
    else:
        # Multiclass case
        if has_proba:
            y_scores = clf.predict_proba(X_test)
        else:
            decision_values = clf.decision_function(X_test)
            if decision_values.ndim > 1:  # OvO case
                y_scores = np.exp(decision_values) / np.sum(np.exp(decision_values), axis=1, keepdims=True)
            else:  # OvR case
                y_scores = np.column_stack([1 - decision_values, decision_values])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        tpr_at_thresholds = dict()
        
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(
                fpr[i], 
                tpr[i], 
                lw=2, 
                label=f'ROC Class {i} (AUC = {roc_auc[i]:.4f})'
            )
            
            # Calculate TPR at specific FPR thresholds for each class
            tpr_at_thresholds[i] = {}
            for threshold in fpr_thresholds:
                idx = np.argmin(np.abs(fpr[i] - threshold))
                tpr_at_thresholds[i][threshold] = tpr[i][idx]
                plt.plot(fpr[i][idx], tpr[i][idx], 'ro')

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test_bin.ravel(), 
            y_scores.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plt.plot(
            fpr["micro"], 
            tpr["micro"],
            label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:.4f})',
            color='deeppink', 
            linestyle=':', 
            linewidth=4
        )
        
        # Calculate TPR at specific FPR thresholds for micro-average
        tpr_at_thresholds["micro"] = {}
        for threshold in fpr_thresholds:
            idx = np.argmin(np.abs(fpr["micro"] - threshold))
            tpr_at_thresholds["micro"][threshold] = tpr["micro"][idx]
            plt.plot(fpr["micro"][idx], tpr["micro"][idx], 'ro', 
                    label=f'Micro-avg TPR at {threshold*100}% FPR = {tpr["micro"][idx]:.3f}')
        
        metrics.update({
            'roc_auc': roc_auc["micro"],
            'fpr': fpr["micro"],
            'tpr': tpr["micro"],
            'tpr_at_thresholds': tpr_at_thresholds
        })

    # Complete the ROC plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{exp_id}-k_{k_value}-{pretext_classifier}-{clf_name}-ROC")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            save_dir, 
            f"{exp_id}-k_{k_value}-{pretext_classifier}-{clf_name}-roc.png"
        ),
        bbox_inches='tight'
    )
    plt.close()

    return metrics

def plot_confusion_matrices(results, k, save_path):
    """Plot and save confusion matrices"""
    y_true = results['true_labels']
    y_pred = results['predictions']
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (k={k:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

class SARCNN(nn.Module):
    def __init__(self):
        super(SARCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

def train_model(model, train_loader, test_loader, device, epochs=60):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            accuracy = evaluate_model(model, test_loader, device)[0]
            print(f'Epoch {epoch}, Test Accuracy: {accuracy:.2f}%')

def setup_directories():
    try:
        # Create base experiment directory
        if not os.path.exists(EXPERIMENT_BASE_DIR):
            os.makedirs(EXPERIMENT_BASE_DIR)

        # Create subdirectories
        subdirs = {
            'MODEL_DIR': MODEL_DIR,
            'DATA_DIR': DATA_DIR,
            'RESULTS_DIR': RESULTS_DIR,
            'FIGURES_DIR': FIGURES_DIR
        }

        for name, path in subdirs.items():
            if not os.path.exists(path):
                os.makedirs(path)

        return True
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        return False
    
class MultiWriter:
    def __init__(self, filename):
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        # Write to the file
        self.log.write(message)
        self.log.flush()

        # Write to the notebook cell output without recursion
        with self.out:
            self.out.append_stdout(message)  # Avoids calling print()

    def flush(self):
        self.log.flush()

if __name__ == "__main__":
    # First ensure directories exist
    if not setup_directories():
        print("Failed to create necessary directories. Exiting.")
        sys.exit(1)

    timestamp = datetime.now().strftime('%Y-%m-%d-%Hh-%Mm-%Ss')
    filename = os.path.join(RESULTS_DIR, f"{EXP_ID}-07.04.1.1.txt")
    print(f"> Output will be saved to: {filename}")
    
    try:
        sys.stdout = MultiWriter(filename)
        real_root = '/mnt/d/SAMPLE_dataset_public/png_images/qpm/real'
        synthetic_root = '/mnt/d/SAMPLE_dataset_public/png_images/qpm/synth'
        # k_values, accuracies, class_accuracies = run_experiment_4_1(real_root, synthetic_root)
        start_time = datetime.now()
        k_values, results = run_self_supervised_experiment(real_root, synthetic_root)
        end_time = datetime.now()

        duration = end_time - start_time
        print(f"\nStart Time: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")  # Show milliseconds
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")      # Show milliseconds
        # Formatting duration as hh:mm:ss.sss
        total_seconds = duration.total_seconds()
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = duration.microseconds // 1000
        print(f"Duration: {int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}")
        print("\n=== Experiment Finished Successfully ===")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Make sure to restore stdout and close the file
        if isinstance(sys.stdout, MultiWriter):
            sys.stdout.log.close()
        sys.stdout = sys.__stdout__
        print(f"=== PROCESS FINISHED: {EXP_ID} ===")