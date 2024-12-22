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

def create_data_split(dataset: SAMPLEDataset, test_elevation: int = 17, k: float = 1.0) -> Tuple[List[Dict], List[Dict]]:
    """
    Create training and test splits following Experiment 4.1 specifications.
    
    Args:
        dataset: The full dataset
        test_elevation: Elevation angle for test data
        k: Fraction of measured data to use in training [0,1]
    
    Returns:
        Tuple of (training_data, test_data)
    """
    # First get test data - only measured data at test_elevation
    test_data = [d for d in dataset.data 
                 if d['elevation'] == test_elevation and d['is_real']]

    # Group remaining data by class
    class_data = defaultdict(list)
    for d in dataset.data:
        if d['elevation'] != test_elevation:
            class_data[d['label']].append(d)

    # Create training data following paper equations
    training_data = []
    for class_label, data in class_data.items():
        # Get measured and synthetic data
        measured = [d for d in data if d['is_real']]
        synthetic = [d for d in data if not d['is_real']]
        
        # Calculate Nm_j (total measured) and Sm_j (measured test samples)
        Nm_j = len([d for d in dataset.data 
                   if d['label'] == class_label and d['is_real']])
        Sm_j = len([d for d in test_data if d['label'] == class_label])
        
        # Calculate Tm_j (measured training samples) per equation 4.2
        Tm_j = int(k * (Nm_j - Sm_j))
        
        if k > 0:
            # Get measured training samples
            training_measured = measured[:Tm_j]
            training_data.extend(training_measured)
            
            # Get matching synthetic samples for removed measured samples
            removed_measured = measured[Tm_j:]
        else:
            # When k=0, we'll use all synthetic data matched to measured data
            removed_measured = measured

        # Find matching synthetic samples
        for m in removed_measured:
            # Find matching synthetic image by matching elevation and azimuth
            matches = [s for s in synthetic 
                      if s['elevation'] == m['elevation'] and 
                      s['azimuth'] == m['azimuth']]
            if matches:
                training_data.append(matches[0])

    if not training_data:
        raise ValueError(f"No training data generated for k={k}. Check dataset composition.")

    # Verify we have data
    print(f"Training data size: {len(training_data)}")
    print(f"Test data size: {len(test_data)}")
    
    return training_data, test_data

def validate_dataset(dataset):
    if not dataset.data:
        raise ValueError("Dataset is empty")
    
    # Verify we have both measured and synthetic data
    measured = [d for d in dataset.data if d['is_real']]
    synthetic = [d for d in dataset.data if not d['is_real']]
    
    if not measured:
        raise ValueError("No measured data found in dataset")
    if not synthetic:
        raise ValueError("No synthetic data found in dataset")

    # Print dataset composition
    print(f"Total dataset size: {len(dataset.data)}")
    print(f"Measured samples: {len(measured)}")
    print(f"Synthetic samples: {len(synthetic)}")

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

def calculate_class_accuracies(true_labels, pred_labels, num_classes=10):
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for true, pred in zip(true_labels, pred_labels):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    class_accuracies = np.where(class_total > 0,
                               100 * class_correct / class_total,
                               0)
    return class_accuracies

def create_distribution_table(dataset, train_data, test_data, k):
    """
    Create a data distribution table similar to Table 6 in the paper.
    
    Args:
        dataset: Full SAMPLEDataset
        train_data: List of training samples
        test_data: List of test samples
        k: Current k value
    
    Returns:
        pandas DataFrame with the distribution table
    """
    # Create empty table structure
    table_data = []
    
    # Get class mapping for index to name
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    for j in range(10):  # 10 classes
        class_name = idx_to_class[j].upper()
        
        # Calculate values for each row
        Nm_j = len([d for d in dataset.data if d['label'] == j and d['is_real']])
        Ns_j = len([d for d in dataset.data if d['label'] == j and not d['is_real']])
        Sm_j = len([d for d in test_data if d['label'] == j])
        Ss_j = 0  # Always 0 as per paper
        Tm_j = len([d for d in train_data if d['label'] == j and d['is_real']])
        Ts_j = len([d for d in train_data if d['label'] == j and not d['is_real']])
        
        # Add row to table
        table_data.append({
            'Class': class_name,
            'j': j + 1,
            'Nm_j': Nm_j,
            'Ns_j': Ns_j,
            'Sm_j': Sm_j,
            'Ss_j': Ss_j,
            'Tm_j': Tm_j,
            'Ts_j': Ts_j
        })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Add totals row
    totals = df.sum(numeric_only=True)
    totals['Class'] = 'Totals'
    totals['j'] = ''
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    
    return df

def run_experiment_4_1(real_root, synthetic_root, test_elevation=17, n_runs=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define k values according to equation 4.4
    k_values = np.arange(0, 1.05, 0.05)
    accuracies = []
    class_accuracies = []
    
    # Create initial dataset for verification
    dataset = SAMPLEDataset(real_root, synthetic_root)
    
    for k in k_values:
        print(f"\nRunning experiment with k={k:.2f}")
        
        # Create data split for distribution table
        train_data, test_data = create_data_split(dataset, test_elevation, k)
        
        # Create and display distribution table
        print(f"\nData Distribution Table for k={k:.2f}:")
        dist_table = create_distribution_table(dataset, train_data, test_data, k)
        print(tabulate(dist_table, headers='keys', tablefmt='grid', showindex=False))
        print("\nVerifying data split totals:")
        print(f"Total training samples: {len(train_data)}")
        print(f"Total test samples: {len(test_data)}")
        
        k_accuracies = []
        k_class_accuracies = []
        
        for run in range(n_runs):
            print(f"\nRun {run + 1}/{n_runs}")
            
            # Create datasets and dataloaders
            train_dataset = copy.deepcopy(dataset)
            train_dataset.data = train_data
            test_dataset = copy.deepcopy(dataset)
            test_dataset.data = test_data
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=16,
                shuffle=True,
                num_workers=4
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=16,
                shuffle=False,
                num_workers=4
            )
            
            # Train and evaluate model
            model = SARCNN().to(device)
            train_model(model, train_loader, test_loader, device)
            accuracy, preds, labels = evaluate_model(model, test_loader, device)
            
            k_accuracies.append(accuracy)
            k_class_accuracies.append(
                calculate_class_accuracies(labels, preds)
            )
            
            print(f"Run {run + 1} Accuracy: {accuracy:.2f}%")
        
        mean_accuracy = np.mean(k_accuracies)
        mean_class_accuracies = np.mean(k_class_accuracies, axis=0)
        
        accuracies.append(mean_accuracy)
        class_accuracies.append(mean_class_accuracies)
        
        print(f"\nk={k:.2f}, Average Accuracy: {mean_accuracy:.2f}%")
    
    # Plot overall accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies)
    plt.xlabel('Fraction of Measured Training Data (k)')
    plt.ylabel('Probability of Correct Classification')
    plt.title('Experiment 4.1 Results')
    plt.grid(True)
    plt.show()
    
    return k_values, accuracies, class_accuracies

if __name__ == "__main__":
    real_root = "/content/SAMPLE_dataset_public/png_images/qpm/real"
    synthetic_root = "/content/SAMPLE_dataset_public/png_images/qpm/synth"

    k_values, accuracies, class_accuracies = run_experiment_4_1(real_root, synthetic_root)