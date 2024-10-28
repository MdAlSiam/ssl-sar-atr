# data_processor.py
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import pickle
from config import *

def load_and_split_data(data_dir, split_index=0):
    """
    Load data and split into train/test sets based on split_index
    split_index: which 20% chunk to use as test set (0-4)
    """
    images_by_class = {}
    labels_by_class = {}
    
    # Load images
    class_folders = sorted(os.listdir(data_dir))
    for class_index, class_folder in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            # Sort files to ensure consistent ordering
            img_files = sorted(os.listdir(class_path))
            
            images_by_class[class_index] = []
            labels_by_class[class_index] = []
            
            for img_file in img_files:
                img_path = os.path.join(class_path, img_file)
                image = load_img(
                    img_path, 
                    target_size=hyperparameters['img_size'], 
                    color_mode=hyperparameters['color_mode']
                )
                image = img_to_array(image) / 255.0
                images_by_class[class_index].append(image)
                labels_by_class[class_index].append(class_index)
    
    # Split each class based on split_index
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    test_size = hyperparameters['test_split_size']
    for class_idx in images_by_class:
        X = np.array(images_by_class[class_idx])
        y = np.array(labels_by_class[class_idx])
        
        # Calculate indices for the test split
        total_samples = len(X)
        chunk_size = int(total_samples * test_size)
        start_idx = split_index * chunk_size
        end_idx = start_idx + chunk_size
        
        # Create mask for test indices
        test_mask = np.zeros(total_samples, dtype=bool)
        test_mask[start_idx:end_idx] = True
        
        # Split data
        test_images.extend(X[test_mask])
        test_labels.extend(y[test_mask])
        train_images.extend(X[~test_mask])
        train_labels.extend(y[~test_mask])
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

def save_processed_data(data_dir, split_index):
    """Save processed data to disk"""
    x_train, y_train, x_test, y_test = load_and_split_data(data_dir, split_index)
    
    # Create save path based on data directory and split index
    base_name = f"{data_dir.split('/')[-2]}_{data_dir.split('/')[-1]}_split_{split_index}"
    save_path = os.path.join(DATA_DIR, base_name)
    
    # Save the data
    with open(f"{save_path}_train.pkl", 'wb') as f:
        pickle.dump((x_train, y_train), f)
    with open(f"{save_path}_test.pkl", 'wb') as f:
        pickle.dump((x_test, y_test), f)
    
    print(f"Saved processed data to {save_path}")
    return save_path

def load_processed_data(base_path):
    """Load processed data from disk"""
    with open(f"{base_path}_train.pkl", 'rb') as f:
        x_train, y_train = pickle.load(f)
    with open(f"{base_path}_test.pkl", 'rb') as f:
        x_test, y_test = pickle.load(f)
    return x_train, y_train, x_test, y_test