# Self-Supervised Learning for SAR Target Recognition

This repository contains the implementation of a novel self-supervised learning framework for Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR), as described in our paper "Self-Supervised Learning for SAR Target Recognition with Multi-Task Pretext Training."

## Overview

Our framework leverages multi-task pretext training to develop robust feature representations directly from measured SAR data, eliminating the dependency on synthetic data. The implementation includes:

1. **Tensorflow/Keras Implementation** (`exp-07/07.04.0.py`): Primary implementation using TensorFlow/Keras with a CNN-based architecture.
2. **PyTorch Implementation** (`exp-07/07.04.1.1.py`): Implementation of Lewis et al's experiment using PyTorch with CUDA-optimized operations.

## Dataset

The code is designed to work with the Synthetic and Measured Paired and Labeled Experiment (SAMPLE) dataset, which can be obtained from:
https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public

The dataset structure should be:
```
SAMPLE_dataset_public/
├── png_images/
│   ├── decibel/
│   │   ├── real/
│   │   └── synth/
│   └── qpm/
│       ├── real/
│       └── synth/
```

## Implementation Details

### 1. TensorFlow/Keras Implementation (`exp-07/07.04.0.py`)

This implementation includes:

- A configurable multi-task pretext training pipeline
- Feature extraction using a CNN backbone
- Multiple downstream classifiers (SVM, XGBoost, Random Forest, Gradient Boosting)
- Comprehensive evaluation metrics including ROC curves and confusion matrices
- Experiment management with automatic logging

#### Key Configuration Parameters:

```python
EXPERIMENT_BASE_DIR = 'exp-07'
EXP_ID = datetime.now().strftime('%Y%m%d-%H%M%S')

hyperparameters = {
    'img_size': (128, 128),
    'color_mode': 'grayscale',
    'test_split_size': 0.2,
    'cnn_filters': [32, 64, 128, 256],
    'batch_size': 32,
    'epochs': 25,
    'initial_learning_rate': 0.001,
    'early_stopping_patience': 3,
}

experiment_parameters = {
    'mode': 'all',  # choices: 'process', 'pretext', 'downstream', 'all'
    'split_index': 4,
    'architecture': 'cnn',
    'classifier': 'all',
    'data_dir': '/path/to/SAMPLE_dataset_public/png_images/qpm/real'
}
```

#### Running the TensorFlow Implementation:

```bash
python exp-07/07.04.0.py
```

### 2. PyTorch Implementation (`exp-07/07.04.1.1.py`)

This implementation offers:

- GPU-optimized operations using CUDA
- Mixed precision training for faster execution
- Automated k-value experiments as per Lewis et al.'s methodology
- Comprehensive visualization of results

#### Key Configuration Parameters:

```python
EXPERIMENT_BASE_DIR = os.path.join('exp-07', f'{EXP_ID}')

hyperparameters = {
    'rf_n_estimators': 100,
    'svm_kernel': 'linear',
    'gb_n_estimators': 100,
    'xgb_n_estimators': 100,
    'batch_size': 16,
    'num_epochs': 60,
    'learning_rate': 0.001,
    'validation_ratio': 0.15,
    'early_stopping_patience': 5
}
```

#### Running the PyTorch Implementation:

```bash
python exp-07/07.04.1.1.py
```

## Pretext Tasks

Both implementations use the following pretext tasks for self-supervised learning:

1. **Original Image Preservation**: The unaltered image to maintain SAR signature characteristics
2. **Rotations (90°, 180°, 270°)**: Different orientation perspectives
3. **Gaussian Blur**: Simulates resolution degradation
4. **Flips (Horizontal, Vertical)**: Creates mirrored versions
5. **Denoising**: Using BM3D algorithm to learn noise-robust features
6. **Zoom-In Transformation**: Creating multi-scale representations

## Results and Evaluation

The code generates:

1. **Training History**: Logs and plots of pretext task training
2. **Performance Metrics**: Accuracy, precision, recall, F1-score
3. **ROC Curves**: With true positive rates at specified false positive rate thresholds
4. **Confusion Matrices**: For downstream classification tasks

Results are saved in:
- `EXPERIMENT_BASE_DIR/results/`: Metrics and logs
- `EXPERIMENT_BASE_DIR/figures/`: Plots and visualizations

## Comparison with Lewis et al.

Our implementation includes a version of the experiment from Lewis et al. (2019) to enable direct comparison:

- In our version, we take only the measured data decreased by more than 30% without synthetic replacement
- In Lewis et al.'s adaptation, we gradually decrease the amount of measured data (k-value), but the missing measured data is not replaced with synthetic equivalents

## License

This code is provided for academic research purposes only. Please cite our paper if you use this code:

```
@article{siam2025selfsupervised,
  title={Self-Supervised Learning for SAR Target Recognition with Multi-Task Pretext Training},
  author={Siam, Md Al and Noor, Dewan Fahim},
  journal={},
  year={2025}
}
```

<!-- ## Acknowledgments

This work is supported by the funds provided by the National Science Foundation and by DoD OUSD (R&E) under Cooperative Agreement PHY-2229929 (The NSF AI Institute for Artificial and Natural Intelligence). -->