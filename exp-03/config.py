# config.py
import os

RANDOM_SEED = 42
MODEL_DIR = os.path.join("pretext_models")
DATA_DIR = os.path.join("processed_data")
RESULTS_DIR = os.path.join("results")

# Create required directories
for dir_path in [MODEL_DIR, DATA_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Hyperparameters configuration
hyperparameters = {
    # Data loading and splitting
    'img_size': (128, 128),
    'color_mode': 'grayscale',
    'test_split_size': 0.2,
    'test_split_index': 0,  # Which 20% chunk to use (0-4)
    
    # Model architecture
    'input_shape': (128, 128, 1),
    'num_classes': 4,
    'fine_tune': True,
    
    # CNN architecture
    'cnn_filters': [32, 64, 128, 256],
    'cnn_kernel_size': (3, 3),
    'cnn_pool_size': (2, 2),
    'cnn_dense_units': 512,
    'cnn_dropout_rate': 0.5,
    
    # Training configuration
    'batch_size': 32,
    'epochs': 10,
    'validation_split': 0.25,
    'learning_rate': 0.001,
    'early_stopping_patience': 3,
    'lr_reduction_patience': 3,
    'lr_reduction_factor': 0.5,
    
    # Downstream task
    'feature_extraction_layer': -2,
    'rf_n_estimators': 100,
    'svm_kernel': 'linear',
    'gb_n_estimators': 100,
    'xgb_n_estimators': 100,
    'cv_splits': 5,
}