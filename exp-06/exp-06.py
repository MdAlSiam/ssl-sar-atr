import os
import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
from datetime import datetime
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from tabulate import tabulate
from typing import Dict, List, Any
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from datetime import timedelta
from IPython.display import display, clear_output
from ipywidgets import Output
import subprocess
import cv2
from scipy.ndimage import gaussian_filter
import bm3d
from enum import Enum
import matplotlib
from PIL import Image
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend

RANDOM_SEED = 42
EXP_ID = datetime.now().strftime('%Y%m%d-%H%M%S')
EXPERIMENT_BASE_DIR = 'exp-06'
MODEL_DIR = os.path.join(EXPERIMENT_BASE_DIR, "pretext_models")
DATA_DIR = os.path.join(EXPERIMENT_BASE_DIR, "processed_data")
RESULTS_DIR = os.path.join(EXPERIMENT_BASE_DIR, "results")
FIGURES_DIR = os.path.join(EXPERIMENT_BASE_DIR, "figures")

class PreextTask(Enum):
    ORIGINAL = 0
    ROTATE_90 = 1
    ROTATE_180 = 2
    ROTATE_270 = 3
    BLUR = 4
    FLIP_HORIZONTAL = 5
    FLIP_VERTICAL = 6
    FLIP_BOTH = 7
    DENOISE = 8
    ZOOM_IN = 9
    # ZOOM_OUT = 10

architecture_list = [
    'cnn',
    'unet',
    'inceptionv3',
    'resnet50',
    'efficientnetb0',
    'resnet101',
    'resnet152',

    # 'vgg16',
    # 'vgg19',
]

classifier_list = [
    'xgboost',
    'gradient_boosting',
    'svm',
    'random_forest',
]

experiment_parameters = {
    'mode': 'all',  # choices: 'process', 'pretext', 'downstream', 'all'
    'split_index': 4,
    'architecture': 'cnn',  # options: specific architecture like 'cnn' or 'all'
    'classifier': 'all',  # choices: 'random_forest', 'svm', 'gradient_boosting', 'xgboost'
    'data_dir': '/mnt/d/SAMPLE_dataset_public/png_images/qpm/real' # Currently working only with the qpm/real folder
}

# Configuration dictionary
hyperparameters = {
    # Data loading and splitting
    'img_size': (128, 128),
    'color_mode': 'grayscale',
    'test_split_size': 0.2,
    'test_split_index': experiment_parameters['split_index'],

    # Model architecture
    'input_shape': (128, 128, 1),
    'num_classes': len(PreextTask),
    'fine_tune': True,

    # CNN architecture
    'cnn_filters': [32, 64, 128, 256],
    'cnn_kernel_size': (3, 3),
    'cnn_pool_size': (2, 2),
    'cnn_dense_units': 512,
    'cnn_dropout_rate': 0.5,
    'cnn_loss_function': 'sparse_categorical_crossentropy',
    'cnn_activation_function': 'relu',
    'cnn_padding': 'same',

    # Other Pretext Architectures
    'pretained_model_weights': 'imagenet',

    # Training configuration
    'batch_size': 32,
    'epochs': 25,
    'validation_split': 0.25,
    'learning_rate': 0.001,
    'early_stopping_patience': 3,
    'lr_reduction_patience': 3,
    'lr_reduction_factor': 0.25,

    # Downstream task
    'feature_extraction_layer': None,
    'rf_n_estimators': 100,
    'svm_kernel': 'linear',
    'gb_n_estimators': 100,
    'xgb_n_estimators': 100,
    'cv_splits': 5,
}

def display_configurations():
    def create_section_table(keys, header):
        table = [[key, hyperparameters[key]] for key in keys]
        return f"\n{header}\n" + tabulate(table, headers=['Parameter', 'Value'], tablefmt='grid')

    print("========== Experiment Configurations ==========")

    # Architectures and Classifiers
    print("\n>>> Architectures <<<")
    print(tabulate([[", ".join(architecture_list)]], tablefmt='grid'))

    print("\n>>> Classifiers <<<")
    print(tabulate([[", ".join(classifier_list)]], tablefmt='grid'))

    # Experiment Pipeline
    print("\n>>> THIS EXPERIMENT PIPELINE <<<")
    table = [[key, value] for key, value in experiment_parameters.items()]
    print(tabulate(table, headers=['Parameter', 'Value'], tablefmt='grid'))

    # Data Loading and Splitting
    data_keys = ['img_size', 'color_mode', 'test_split_size', 'test_split_index']
    print(create_section_table(data_keys, ">>> Data Loading and Splitting <<<"))

    # Model Architecture
    model_keys = ['input_shape', 'num_classes', 'fine_tune']
    print(create_section_table(model_keys, ">>> Model Architecture <<<"))

    # CNN Architecture
    cnn_keys = ['cnn_filters', 'cnn_kernel_size', 'cnn_pool_size', 'cnn_dense_units',
                'cnn_dropout_rate', 'cnn_loss_function', 'cnn_activation_function', 'cnn_padding']
    print(create_section_table(cnn_keys, ">>> CNN Architecture <<<"))

    # Other Pretext Architectures
    pretext_keys = ['pretained_model_weights']
    print(create_section_table(pretext_keys, ">>> Other Pretext Architectures <<<"))

    # Training Configuration
    training_keys = ['batch_size', 'epochs', 'validation_split', 'learning_rate',
                    'early_stopping_patience', 'lr_reduction_patience', 'lr_reduction_factor']
    print(create_section_table(training_keys, ">>> Training Configuration <<<"))

    # Downstream Task Configuration
    downstream_keys = ['feature_extraction_layer', 'rf_n_estimators', 'svm_kernel',
                      'gb_n_estimators', 'xgb_n_estimators', 'cv_splits']
    print(create_section_table(downstream_keys, ">>> Downstream Task Configuration <<<"))

def get_cuda_version():
    try:
        # Run the nvcc command to get the CUDA version
        cuda_version = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        return cuda_version
    except Exception as e:
        return f"Error getting CUDA version: {e}"

def get_nvidia_smi_info():
    try:
        # Run the nvidia-smi command to get GPU information
        nvidia_smi_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        return nvidia_smi_info
    except Exception as e:
        return f"Error getting nvidia-smi info: {e}"

def check_environment():
    """Print and check library versions against requirements"""

    # Import all required libraries
    import sys
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost as xgb
    import matplotlib
    from PIL import Image
    import tabulate as tabulate_module  # Renamed to avoid confusion

    # Define required versions
    requirements = {
        'tensorflow': '2.8.0',
        'numpy': '1.19.2',
        'scikit-learn': '0.24.2',
        'matplotlib': '3.4.3',
        'xgboost': '1.5.0',
        'Pillow': '8.3.2',
        'pandas': '2.2.3',
        'tabulate': '0.9.0',
        'ipython': '7.31.1',
        'ipykernel': '6.29.5',
        'ipywidgets': '8.1.5',
        'jupyter_client': '8.6.3',
        'jupyter_core': '5.7.2',
        'jupyter_server': '2.14.2',
        'jupyterlab': '4.2.5',
        'nbclient': '0.10.0',
        'nbconvert': '7.16.4',
        'nbformat': '5.10.4',
        'notebook': '7.2.2',
        'traitlets': '5.14.3'
    }

    # Initialize table data
    table_data = []

    # Check core libraries with better error handling
    try:
        libraries_check = [
            ('Python', sys.version.split()[0], 'N/A'),
            ('TensorFlow', tf.__version__, requirements['tensorflow']),
            ('NumPy', np.__version__, requirements['numpy']),
            ('Pandas', pd.__version__, requirements['pandas']),
            ('Scikit-learn', sklearn.__version__, requirements['scikit-learn']),
            ('XGBoost', xgb.__version__, requirements['xgboost']),
            ('Matplotlib', matplotlib.__version__, requirements['matplotlib']),
            ('Pillow (PIL)', Image.__version__, requirements['Pillow']),
        ]

        # Add tabulate version separately with error handling
        try:
            tabulate_version = getattr(tabulate_module, '__version__', 'Version unknown')
        except:
            tabulate_version = 'Version unknown'
        libraries_check.append(('Tabulate', tabulate_version, requirements['tabulate']))

    except Exception as e:
        print(f"Error checking core library versions: {str(e)}")
        return

    # Add Jupyter-related libraries
    jupyter_libs = {
        'IPython': 'ipython',
        'IPykernel': 'ipykernel',
        'IPywidgets': 'ipywidgets',
        'Jupyter Client': 'jupyter_client',
        'Jupyter Core': 'jupyter_core',
        'Jupyter Server': 'jupyter_server',
        'JupyterLab': 'jupyterlab',
        'NBClient': 'nbclient',
        'NBConvert': 'nbconvert',
        'NBFormat': 'nbformat',
        'Notebook': 'notebook',
        'Traitlets': 'traitlets'
    }

    for display_name, pkg_name in jupyter_libs.items():
        try:
            pkg = __import__(pkg_name)
            version = getattr(pkg, '__version__', 'Version unknown')
            req_version = requirements.get(pkg_name, 'N/A')
            libraries_check.append((display_name, version, req_version))
        except ImportError:
            libraries_check.append((display_name, 'Not installed', requirements.get(pkg_name, 'N/A')))
        except Exception as e:
            libraries_check.append((display_name, f'Error: {str(e)}', requirements.get(pkg_name, 'N/A')))

    # Create table with version comparison and status
    for name, current, required in libraries_check:
        if required == 'N/A':
            status = '---'
        elif current == 'Not installed' or 'Error' in str(current) or current == 'Version unknown':
            status = '?'
        else:
            status = '✓' if current == required else '✗'

        table_data.append([name, current, required, status])

    # Print main version table
    print("\n" + "="*100)
    print(f"{' LIBRARY VERSIONS AND REQUIREMENTS ':=^100}")
    print("="*100)
    print(tabulate_module.tabulate(table_data,
                                 headers=['Library', 'Current Version', 'Required Version', 'Status'],
                                 tablefmt='grid'))

    # Print system and GPU information
    print("\n" + "="*100)
    print(f"{' SYSTEM INFORMATION ':=^100}")
    print("="*100)

    # System info
    import platform
    print(f"OS: {platform.system()} {platform.version()}")

    # GPU info
    print("\nGPU Information:")
    print(f"CUDA Available: {tf.test.is_built_with_cuda()}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  • {gpu}")
    else:
        print("No GPUs found")

    # Print CUDA version
    print("CUDA Version:")
    print(get_cuda_version())

    # Print NVIDIA GPU information
    print("NVIDIA GPU Info:")
    print(get_nvidia_smi_info())

    # TensorFlow environment variables
    tf_vars = [var for var in os.environ if 'TF_' in var]
    if tf_vars:
        print("TensorFlow Environment Variables:")
        for var in tf_vars:
            print(f"  • {var}: {os.environ[var]}")

    print("="*100 + "\n")

def set_all_seeds(seed=42, gpu_deterministic=False):
    """Set all seeds to make results reproducible
    Args:
        seed (int): The seed value to use
        gpu_deterministic (bool): Whether to enable deterministic GPU operations
            Set to False for better performance, True for reproducibility
    """
    # Basic seeds
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.experimental.numpy.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    if gpu_deterministic:
        print("> Warning: Enabling deterministic GPU operations may impact performance")
        # GPU-specific deterministic settings
        tf.config.experimental.enable_op_determinism()
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        # Threading controls
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    else:
        print("> Using non-deterministic GPU operations for better performance")
        # Disable deterministic operations
        if 'TF_DETERMINISTIC_OPS' in os.environ:
            del os.environ['TF_DETERMINISTIC_OPS']
        if 'TF_CUDNN_DETERMINISTIC' in os.environ:
            del os.environ['TF_CUDNN_DETERMINISTIC']

        try:
            tf.config.experimental.disable_op_determinism()
        except:
            pass  # Ignore if the function doesn't exist in current TF version

class ResultsTracker:
    def __init__(self):
        self.results = []

    def add_result(self, data_dir: str, architecture: str, classifier: str,
                   cv_metrics: Dict[str, List[float]], test_metrics: Dict[str, Any]):
        result = {
            'Dataset': f"{data_dir.split('/')[-2]}_{data_dir.split('/')[-1]}_test_split_ind_{hyperparameters['test_split_index']}",
            'Architecture': architecture,
            'Classifier': classifier,
            'CV_Accuracy': f"{np.mean(cv_metrics['accuracies'])*100:.1f}±{np.std(cv_metrics['accuracies'])*100:.1f}",
            'CV_Precision': f"{np.mean(cv_metrics['precisions'])*100:.1f}±{np.std(cv_metrics['precisions'])*100:.1f}",
            'CV_Recall': f"{np.mean(cv_metrics['recalls'])*100:.1f}±{np.std(cv_metrics['recalls'])*100:.1f}",
            'CV_F1': f"{np.mean(cv_metrics['f1_scores'])*100:.1f}±{np.std(cv_metrics['f1_scores'])*100:.1f}",
            'Test_Accuracy': f"{test_metrics['accuracy']*100:.1f}",
            'Test_Precision': f"{test_metrics['precision']*100:.1f}",
            'Test_Recall': f"{test_metrics['recall']*100:.1f}",
            'Test_F1': f"{test_metrics['f1']*100:.1f}",
            'Test_Confusion_Matrix': f"{test_metrics['confusion_matrix']}"
        }
        self.results.append(result)

    def display_results(self):
        if not self.results:
            print("> No results to display")
            return

        df = pd.DataFrame(self.results)
        grouped = df.groupby(['Dataset', 'Architecture'])

        for (dataset, arch), group in grouped:
            print(f"\n=== Results: Dataset: <{dataset}> | Pretext Model: <{arch}> ===")

            # Display test and cross-validation metrics
            display_cols = [
                'Classifier',
                'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1',
                'CV_Accuracy', 'CV_Precision', 'CV_Recall', 'CV_F1'
            ]

            display_df = group[display_cols].copy()
            print("\nMetrics Summary:")
            print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))

            # Separate classifiers and confusion matrices
            classifier_names = group['Classifier'].tolist()
            confusion_matrices = group['Test_Confusion_Matrix'].tolist()

            # Build display format for classifiers and confusion matrices
            display_confusion_matrix = pd.DataFrame([confusion_matrices], columns=classifier_names, index=["Confusion Matrix"])
            print(tabulate(display_confusion_matrix, headers='keys', tablefmt='grid', showindex=False))

def load_and_split_data(data_dir, split_index=0):
    images_by_class = {}
    labels_by_class = {}

    class_folders = sorted(os.listdir(data_dir))
    print(f'> [load_and_split_data] Data preparing from the class: ')
    for class_index, class_folder in enumerate(class_folders):
        print(f'[{class_index} {class_folder}]', end=' ')
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            img_files = sorted(os.listdir(class_path))[:92] # TODO For now pick the first 92, will choose randomly later

            images_by_class[class_index] = []
            labels_by_class[class_index] = []

            for img_file in img_files:
                img_path = os.path.join(class_path, img_file)
                image = load_img(
                    img_path,
                    target_size=hyperparameters['img_size'],
                    color_mode=hyperparameters['color_mode']
                )
                image = img_to_array(image) / 255.0 # TODO Try baseline scaling
                images_by_class[class_index].append(image)
                labels_by_class[class_index].append(class_index)
    print()

    train_images, train_labels = [], []
    test_images, test_labels = [], []

    print("\n=== [DATA DISTRIBUTION] ===")
    print(f"{'Class':^10} {'Total':^10} {'Train':^10} {'Test':^10} {'Start':^10} {'End':^10}")
    print("-" * 60)

    test_size = hyperparameters['test_split_size']
    for class_idx in sorted(images_by_class.keys()):
        X = np.array(images_by_class[class_idx])
        y = np.array(labels_by_class[class_idx])

        total_samples = len(X)
        chunk_size = int(total_samples * test_size)
        start_idx = split_index * chunk_size
        end_idx = start_idx + chunk_size

        test_mask = np.zeros(total_samples, dtype=bool)
        test_mask[start_idx:end_idx] = True

        test_images.extend(X[test_mask])
        test_labels.extend(y[test_mask])
        train_images.extend(X[~test_mask])
        train_labels.extend(y[~test_mask])

        print(f"{class_idx:^10} {total_samples:^10} {sum(~test_mask):^10} {sum(test_mask):^10} {start_idx:^10} {end_idx-1:^10}")

    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

def save_processed_data(data_dir, split_index):
    base_name = f"{data_dir.split('/')[-2]}_{data_dir.split('/')[-1]}_split_{split_index}"
    save_path = os.path.join(DATA_DIR, base_name)

    if os.path.exists(f"{save_path}_train.pkl"):
        print(f"> Data already saved in {save_path} (_train.pkl or _test.pkl)")
        return

    x_train, y_train, x_test, y_test = load_and_split_data(data_dir, split_index)

    with open(f"{save_path}_train.pkl", 'wb') as f:
        pickle.dump((x_train, y_train), f)
    with open(f"{save_path}_test.pkl", 'wb') as f:
        pickle.dump((x_test, y_test), f)

    print(f"> Saved processed data to {save_path}")
    return save_path

def load_processed_data(base_path):
    with open(f"{base_path}_train.pkl", 'rb') as f:
        x_train, y_train = pickle.load(f)
    with open(f"{base_path}_test.pkl", 'rb') as f:
        x_test, y_test = pickle.load(f)

    test_size = hyperparameters['test_split_size']
    print("=== Summary of Loaded Data: ===")
    print(f"Total images: {len(x_train) + len(x_test)}")
    print(f"Training set: {len(x_train)} images ({(1-test_size)*100:.0f}%)")
    print(f"Test set: {len(x_test)} images ({test_size*100:.0f}%)")
    print(f"Image shape: {x_train[0].shape}")
    print()

    return x_train, y_train, x_test, y_test

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
        
        # Create empty image with original size
        # padded = np.zeros_like(image)
        # padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = zoomed
        # zoomed = padded
        # Use cv2.copyMakeBorder for border replication
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
        
    elif pretext_task == PreextTask.FLIP_BOTH:
        aug_image = cv2.flip(aug_image, -1)
        
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

def preprocess_data(images):
    """
    Preprocess images for pretext tasks in representation learning
    """
    augmented_images = []
    pretext_labels = []
    original_indices = []
    
    for idx, image in enumerate(images):
        # Ensure input image has correct shape
        if len(image.shape) == 2:
            image = image[..., np.newaxis]  # Add channel dimension if missing
            
        for task in PreextTask:
            aug_image, task_label = augment_single_image(image, task)
            # Ensure augmented image maintains shape consistency
            if len(aug_image.shape) == 2:
                aug_image = aug_image[..., np.newaxis]
            augmented_images.append(aug_image)
            pretext_labels.append(task_label)
            original_indices.append(idx)
    
    # Stack arrays instead of using np.array to ensure consistent shapes
    augmented_images = np.stack(augmented_images, axis=0)
    pretext_labels = np.array(pretext_labels)
    original_indices = np.array(original_indices)
    
    print(f'> Generated {len(augmented_images)} images of shape {augmented_images[0].shape} with {len(pretext_labels)} labels')
    print(f'> Number of pretext tasks (classes): {len(PreextTask)}')
    
    return augmented_images, pretext_labels

def print_training_history(history):
    """Print training history with timing information"""

    # Create list of lists with epoch number and timing
    data: List[List[Union[int, float]]] = []  # Explicitly type the data list
    total_time: float = 0.0

    for epoch in range(len(history.history['loss'])):
        # Initialize row with explicit types
        current_row: List[Union[int, float]] = []

        # Add epoch number (integer)
        current_row.append(int(epoch + 1))

        # Add metrics (floats)
        for metric in history.history.keys():
            if metric not in ['batch', 'size', 'time']:
                current_row.append(float(history.history[metric][epoch]))

        # Add timing information if available
        if hasattr(history, 'epoch_times'):
            epoch_time = float(history.epoch_times[epoch])
            total_time += epoch_time
            current_row.append(float(epoch_time))
            current_row.append(float(total_time))

        data.append(current_row)

    # Create headers
    headers: List[str] = ['Epoch']
    for key in history.history.keys():
        if key not in ['batch', 'size', 'time']:
            headers.append(key.replace('_', ' ').title())
    if hasattr(history, 'epoch_times'):
        headers.extend(['Time (s)', 'Total Time (s)'])

    # Print summary header
    print("\n" + "="*80)
    print(f"{' TRAINING HISTORY ':=^80}")
    print("="*80)

    # Print table
    print(tabulate(data,
                  headers=headers,
                  floatfmt=('.0f', '.4f', '.4f', '.4f', '.4f', '.3f', '.3f'),
                  tablefmt='grid'))

    # Print timing summary if available
    if hasattr(history, 'epoch_times'):
        avg_time = total_time / len(history.epoch_times)
        total_time_td = timedelta(seconds=total_time)
        avg_time_td = timedelta(seconds=avg_time)
        print("\nTiming Summary:")
        print(f"  • Total Training Time: {total_time_td} (hh:mm:ss)")
        print(f"  • Average Epoch Time: {avg_time_td} (hh:mm:ss)")
        print("="*80 + "\n")

def print_model_summary(model, method='compact'):
    """Print model summary in different formats"""
    if method == 'compact':
        model_name = model.name
        model_type = model.__class__.__name__

        print("=== Model Architecture ===")
        print(f"Model: \"{model_name}\" ({model_type})")

        table_data = []
        
        # Get connectivity information from the model
        model_config = model.get_config()
        layer_connections = {}
        
        # Process inbound nodes from model config
        if 'layers' in model_config:
            for layer_config in model_config['layers']:
                layer_name = layer_config['config']['name']
                inbound_nodes = layer_config.get('inbound_nodes', [])
                if inbound_nodes:
                    layer_connections[layer_name] = [node[0][0] for node in inbound_nodes]
                else:
                    layer_connections[layer_name] = ['None']

        # Process each layer
        for layer in model.layers:
            # Get layer type
            layer_type = layer.__class__.__name__

            # Get output shape
            if hasattr(layer, 'output_shape'):
                output_shape = str(layer.output_shape)
            else:
                output_shape = 'unknown'

            # Get parameters
            try:
                params = layer.count_params()
                trainable = layer.trainable
                trainable_p = sum([tf.size(w).numpy() for w in layer.trainable_weights]) if layer.trainable_weights else 0
                non_trainable_p = sum([tf.size(w).numpy() for w in layer.non_trainable_weights]) if layer.non_trainable_weights else 0
            except:
                params = 0
                trainable = False
                trainable_p = 0
                non_trainable_p = 0

            # Get connected to information
            connected_to = layer_connections.get(layer.name, ['None'])
            connected_to_str = ', '.join(connected_to)

            table_data.append([
                layer.name,
                layer_type,
                output_shape,
                connected_to_str,
                f"{params:,}",
                f"{trainable_p:,}",
                f"{non_trainable_p:,}",
                "✓" if trainable else "✗"
            ])

        # Print the table
        headers = ['Layer', 'Type', 'Output Shape', 'Connected to', 'Params', 'Trainable P', 'Non-trainable P', 'Trainable']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))

    elif method == 'detailed':
        # Use Keras's built-in summary method
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        print("\n" + "="*80)
        print(f"{' DETAILED MODEL SUMMARY ':=^80}")
        print("="*80)
        print('\n'.join(stringlist))
        print("="*80 + "\n")

def build_unet(inputs):
    """
    Build U-Net model with attention mechanisms
    """
    # Initial feature extraction
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Store skip connections
    skips = []

    # Encoding path
    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        skips.append(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)

    # Bridge
    x = layers.Conv2D(1024, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(1024, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Decoding path
    for filters, skip in zip([512, 256, 128, 64], reversed(skips)):
        x = layers.UpSampling2D((2, 2))(x)
        
        # Attention gate
        attention_gate = layers.multiply([
            x,
            layers.Conv2D(x.shape[-1], (1, 1), activation='sigmoid')(skip)
        ])
        
        x = layers.Concatenate()([x, attention_gate])
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
    
    return x

def build_basic_cnn(inputs):
    """
    Build a basic CNN model based on hyperparameters
    """
    x = layers.Conv2D(hyperparameters['cnn_filters'][0], hyperparameters['cnn_kernel_size'],
                      activation=hyperparameters['cnn_activation_function'], 
                      padding=hyperparameters['cnn_padding'])(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(hyperparameters['cnn_pool_size'])(x)

    x = layers.Conv2D(hyperparameters['cnn_filters'][1], hyperparameters['cnn_kernel_size'],
                      activation=hyperparameters['cnn_activation_function'], 
                      padding=hyperparameters['cnn_padding'])(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(hyperparameters['cnn_pool_size'])(x)

    x = layers.Conv2D(hyperparameters['cnn_filters'][2], hyperparameters['cnn_kernel_size'],
                      activation=hyperparameters['cnn_activation_function'], 
                      padding=hyperparameters['cnn_padding'])(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(hyperparameters['cnn_pool_size'])(x)

    x = layers.Conv2D(hyperparameters['cnn_filters'][3], hyperparameters['cnn_kernel_size'],
                      activation=hyperparameters['cnn_activation_function'], 
                      padding=hyperparameters['cnn_padding'])(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Dense(hyperparameters['cnn_dense_units'], 
                    activation=hyperparameters['cnn_activation_function'])(x)
    x = layers.Dropout(hyperparameters['cnn_dropout_rate'])(x)
    
    return x

def build_custom_model(input_shape=(128, 128, 1), num_classes=4, architecture_name='cnn', fine_tune=True):
    """
    Main function to build various models including CNN and U-Net
    """
    # Create input layer
    inputs = tf.keras.Input(shape=input_shape, name='main_input')
    
    if architecture_name == 'cnn':
        x = build_basic_cnn(inputs)
        x = GlobalAveragePooling2D(name='gap_cnn')(x)
    elif architecture_name == 'unet':
        x = build_unet(inputs)
        x = GlobalAveragePooling2D(name='gap_unet')(x)
    else:
        # Handle input size requirements
        required_size = (75, 75) if architecture_name == 'inceptionv3' else (input_shape[0], input_shape[1])
        
        # Preprocessing layers
        x = layers.Resizing(required_size[0], required_size[1], name='resizing')(inputs)
        
        if input_shape[-1] == 1:
            x = layers.Conv2D(3, (1, 1), name='channel_expansion')(x)

        # Base model selection and configuration
        base_model_map = {
            'resnet50': (tf.keras.applications.ResNet50, (224, 224)),
            'resnet101': (tf.keras.applications.ResNet101, (224, 224)),
            'resnet152': (tf.keras.applications.ResNet152, (224, 224)),
            'efficientnetb0': (tf.keras.applications.EfficientNetB0, (224, 224)),
            'vgg16': (tf.keras.applications.VGG16, (224, 224)),
            'vgg19': (tf.keras.applications.VGG19, (224, 224)),
            'inceptionv3': (tf.keras.applications.InceptionV3, (299, 299))
        }
        
        if architecture_name not in base_model_map:
            raise ValueError(f"Unknown architecture: {architecture_name}")
            
        base_model_class, min_size = base_model_map[architecture_name]
        
        # Ensure minimum size requirements are met
        if required_size[0] < min_size[0] or required_size[1] < min_size[1]:
            x = layers.Resizing(min_size[0], min_size[1], name='size_adjustment')(x)
            
        # Create and configure base model
        base_model = base_model_class(
            include_top=False,
            weights=hyperparameters['pretained_model_weights'],
            input_shape=(min_size[0], min_size[1], 3)
        )
        
        if not fine_tune:
            base_model.trainable = False
            
        # Apply base model and global pooling
        x = base_model(x, training=fine_tune)
        x = GlobalAveragePooling2D(name='gap_transfer')(x)

    # Final classification layer
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create and compile model
    model = models.Model(inputs=inputs, outputs=outputs, name=f'{architecture_name}_model')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
        loss=hyperparameters['cnn_loss_function'], # 'cnn_loss_function': 'sparse_categorical_crossentropy'
        metrics=['accuracy']
    )

    print_model_summary(model=model)

    return model

def save_model(model, model_path):
    try:
        try:
            model.save(model_path, save_format='h5')
        except Exception as h5_error:
            print(f"> H5 saving failed, trying SavedModel format: {h5_error}")
            model_path = model_path.replace('.h5', '')
            model.save(model_path, save_format='tf')

        print(f"> Model successfully saved at: {model_path}")
        return model_path
    except Exception as e:
        print(f"> Error saving model: {str(e)}")
        raise

def load_model(architecture_name, data_path, split_index):
    model_name = f"pretext_model_{os.path.basename(data_path)}_{architecture_name}.h5"
    model_path = os.path.join(MODEL_DIR, model_name)

    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        else:
            model_path = model_path.replace('.h5', '')
            model = tf.keras.models.load_model(model_path)
        print(f"> Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        print(f"\n> Error loading model: {str(e)}")
        raise

def run_pretext_pipeline(x_augmented, y_augmented, architecture_name, model_path):
    model = build_custom_model(
        input_shape=hyperparameters['input_shape'],
        num_classes=hyperparameters['num_classes'],
        architecture_name=architecture_name,
        fine_tune=hyperparameters['fine_tune']
    )

    class TimingCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.epoch_times = []
            self.epoch_start_time = None

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = datetime.now()

        def on_epoch_end(self, epoch, logs=None):
            epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
            self.epoch_times.append(epoch_time)

    timing_callback = TimingCallback()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=hyperparameters['early_stopping_patience'],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=hyperparameters['lr_reduction_factor'],
            patience=hyperparameters['lr_reduction_patience']
        ),
        timing_callback 
    ]

    history = model.fit(
        x_augmented, y_augmented,
        validation_split=hyperparameters['validation_split'],
        batch_size=hyperparameters['batch_size'],
        epochs=hyperparameters['epochs'],
        callbacks=callbacks,
        shuffle=True,
        verbose=0
    )

    # Add timing information to history object
    history.epoch_times = timing_callback.epoch_times

    # Save the trained model
    save_model(model, model_path)

    return history

def create_layer_inspector(model):
    """
    Creates a function to inspect intermediate layer outputs
    
    Args:
        model: The Keras model to inspect
        
    Returns:
        inspect_layer: Function that takes input data and layer name(s) and returns their outputs
    """
    def inspect_layer(input_data, layer_names):
        """
        Get output from specific layers
        
        Args:
            input_data: Input data to run through the model
            layer_names: String or list of strings of layer names to inspect
            
        Returns:
            Dictionary of layer names and their outputs
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]
            
        # Create inspection models for each layer
        inspection_models = {}
        for layer_name in layer_names:
            try:
                layer = model.get_layer(layer_name)
                inspection_models[layer_name] = tf.keras.Model(
                    inputs=model.input,
                    outputs=layer.output,
                    name=f"inspection_{layer_name}"
                )
            except ValueError as e:
                # print(f"Error: Layer '{layer_name}' not found. Available layers:")
                # for idx, layer in enumerate(model.layers):
                #     print(f"{idx}: {layer.name}")
                raise e
        
        # Get outputs
        outputs = {}
        for layer_name, inspection_model in inspection_models.items():
            outputs[layer_name] = inspection_model.predict(input_data, verbose=2)
            
        return outputs

    return inspect_layer

def extract_features(pretext_model, x_data, layer_name=None, verbose=True):
    """
    Extracts and analyzes features from a specific layer of the model using layer inspector.
    
    Args:
        pretext_model: The keras model to extract features from
        x_data: Input data to process
        layer_name: Name of the layer to extract features from. 
                   If None, uses the second to last layer
        verbose: Whether to print detailed statistics about the extracted features
    
    Returns:
        numpy array of extracted features
    """
    # Create layer inspector
    inspector = create_layer_inspector(pretext_model)
    
    # If no layer name provided, use second to last layer
    if layer_name is None:
        layer_name = pretext_model.layers[-2].name
        print(f"> No layer name provided. Using second to last layer: '{layer_name}'")
    
    if verbose:
        print(f"Available layers ({len(pretext_model.layers)}):")
        # for idx, layer in enumerate(pretext_model.layers):
        #     print(f"{idx}: {layer.name}")
    
    try:
        # Use inspector to get layer outputs
        outputs = inspector(x_data, layer_name)
        features = outputs[layer_name]
        
        if verbose:
            # Print basic statistics
            print(f"\n{layer_name} output statistics:")
            print(f"Shape: {features.shape}")
            print(f"Range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"Mean: {features.mean():.3f}")
            print(f"Std: {features.std():.3f}")
            
            # Additional analysis for specific layer types
            if 'conv' in layer_name.lower():
                print("\nChannel statistics:")
                for i in range(features.shape[-1]):
                    channel_data = features[..., i]
                    print(f"Channel {i}: Mean={channel_data.mean():.3f}, Std={channel_data.std():.3f}")
            
            elif 'batch_normalization' in layer_name.lower():
                print("\nBatch normalization stats (should be roughly mean=0, std=1):")
                print(f"Global Mean: {features.mean():.3f}")
                print(f"Global Std: {features.std():.3f}")
        
        return features
        
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        raise

def evaluate_downstream_task(clf, X_test, y_test, pretext_classifier, save_dir=FIGURES_DIR):
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
    
    # ROC curve analysis
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
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        
        metrics.update({
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        })
    else:
        # Multiclass case
        if has_proba:
            y_scores = clf.predict_proba(X_test)
        else:
            # For decision_function with multiclass, we need to handle OvR and OvO cases
            decision_values = clf.decision_function(X_test)
            if decision_values.ndim == 1:
                # OvR case - reshape to match predict_proba format
                y_scores = np.column_stack([1 - decision_values, decision_values])
            else:
                # OvO case - convert to OvR probabilities
                y_scores = np.zeros((len(y_test), n_classes))
                for i in range(n_classes):
                    y_scores[:, i] = (decision_values[:, i] > 0).mean(axis=1)
                
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(
                fpr[i], 
                tpr[i], 
                lw=2, 
                label=f'ROC curve class {i} (AUC = {roc_auc[i]:.2f})'
            )
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test_bin.ravel(), 
            y_scores.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plt.plot(
            fpr["micro"], 
            tpr["micro"],
            label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
            color='deeppink', 
            linestyle=':', 
            linewidth=4
        )
        
        metrics.update({
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        })
    
    # Complete the ROC plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{EXP_ID}-sp_{hyperparameters['test_split_index']}-{pretext_classifier}-{clf_name}-ROC")
    plt.legend(loc="lower right")
    
    # Save the plot if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                save_dir, 
                f"{EXP_ID}-sp_{hyperparameters['test_split_index']}-{pretext_classifier}-{clf_name}-roc.png"
            )
        )
    plt.close()
    
    return metrics

def train_downstream_task(train_features, train_labels, test_features, test_labels, classifier='random_forest', n_splits=None, pretext_classifier='cnn'):
    # Assign number of splits from hyperparameters if not specified
    if n_splits is None:
        n_splits = hyperparameters['cv_splits']

    # Initialize classifier based on hyperparameters
    if classifier == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=hyperparameters['rf_n_estimators'],
            random_state=RANDOM_SEED,
            verbose=0,
        )
    elif classifier == 'svm':
        clf = SVC(
            kernel=hyperparameters['svm_kernel'],
            random_state=RANDOM_SEED,
            verbose=False,
            probability=True
        )
    elif classifier == 'gradient_boosting':
        clf = GradientBoostingClassifier(
            n_estimators=hyperparameters['gb_n_estimators'],
            random_state=RANDOM_SEED,
            verbose=0
        )
    elif classifier == 'xgboost':
        clf = XGBClassifier(
            n_estimators=hyperparameters['xgb_n_estimators'],
            random_state=RANDOM_SEED,
            use_label_encoder=False,  # Update for recent versions of XGBoost
            verbose=0
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = {
        'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': []
    }

    for train_idx, val_idx in skf.split(train_features, train_labels):
        X_train_fold, X_val_fold = train_features[train_idx], train_features[val_idx]
        y_train_fold, y_val_fold = train_labels[train_idx], train_labels[val_idx]

        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_val_fold)

        # Append cross-validation scores
        cv_scores['accuracies'].append(accuracy_score(y_val_fold, y_pred))
        cv_scores['precisions'].append(precision_score(y_val_fold, y_pred, average='macro'))
        cv_scores['recalls'].append(recall_score(y_val_fold, y_pred, average='macro'))
        cv_scores['f1_scores'].append(f1_score(y_val_fold, y_pred, average='macro'))

    # Train final classifier on full training data and evaluate on test data
    clf.fit(train_features, train_labels)
    test_metrics = evaluate_downstream_task(clf, test_features, test_labels, pretext_classifier)

    return clf, cv_scores, test_metrics

def run_downstream_pipeline(train_features, y_train, test_features, y_test, downstream_classifier, pretext_classifier):
    clf, cv_scores, test_metrics = train_downstream_task(
        train_features, y_train,
        test_features, y_test,
        classifier=downstream_classifier,
        n_splits=hyperparameters['cv_splits'],
        pretext_classifier=pretext_classifier
    )

    return clf, cv_scores, test_metrics

def main():
    # Parameters
    mode = experiment_parameters['mode']
    split_index = experiment_parameters['split_index']
    architecture = experiment_parameters['architecture']
    classifier = experiment_parameters['classifier']

    # Data directory
    data_dir = experiment_parameters['data_dir']

    try:
        # Step 1: Process and save data
        if mode in ['process', 'all']:
            print("\n=== Processing Data ===")
            data_path = save_processed_data(data_dir, split_index)

        # Step 2: Train pretext model
        if mode in ['pretext', 'all']:
            print("\n=== Training Pretext Model ===")
            data_path = os.path.join(DATA_DIR, f"{data_dir.split('/')[-2]}_{data_dir.split('/')[-1]}_split_{split_index}")
            x_train, _, _, _ = load_processed_data(data_path)
            x_augmented, y_augmented = preprocess_data(x_train)

            architectures = architecture_list if architecture == 'all' else [architecture]
            for arch in architectures:
                model_name = f"pretext_model_{os.path.basename(data_path)}_{arch}.h5"
                model_path = os.path.join(MODEL_DIR, model_name)

                if os.path.exists(model_path):
                    print(f"> Alredy exists in {model_path}. No need for training.\n")
                    continue

                print(f">> Begin training architecture: {arch}")
                history = run_pretext_pipeline(x_augmented, y_augmented, arch, model_path)
                print_training_history(history)
                # plot_training_history(history)
                print(f">> End training architecture: {arch}\n")

            del x_train
            del x_augmented
            del y_augmented
            gc.collect()

        # Step 3: Run downstream task
        if mode in ['downstream', 'all']:
            print("\n=== Running Downstream Task ===")
            data_path = os.path.join(DATA_DIR, f"{data_dir.split('/')[-2]}_{data_dir.split('/')[-1]}_split_{split_index}")
            x_train, y_train, x_test, y_test = load_processed_data(data_path)

            results_tracker = ResultsTracker()
            architectures = architecture_list if architecture == 'all' else [architecture]
            classifiers = classifier_list if classifier == 'all' else [classifier]

            for arch in architectures:
                pretext_model = load_model(arch, data_path, split_index)
                train_features = extract_features(pretext_model, x_train, hyperparameters['feature_extraction_layer'])
                test_features = extract_features(pretext_model, x_test, hyperparameters['feature_extraction_layer'])
                print()

                for clf in classifiers:
                    print(f">> [STARTED] classifier: {clf} with architecture: {arch}", end=' ')
                    classifier_model, cv_scores, test_metrics = run_downstream_pipeline(
                        train_features, y_train, test_features, y_test, clf, arch
                    )

                    results_tracker.add_result(
                        data_dir=data_dir,
                        architecture=arch,
                        classifier=clf,
                        cv_metrics=cv_scores,
                        test_metrics=test_metrics
                    )
                    print(f"[FINISHED]")

                # results_tracker.display_results()
                del pretext_model
                del train_features
                del test_features
                gc.collect()

            del x_train
            del x_test
            del y_train
            del y_test
            gc.collect()

            results_tracker.display_results()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

class MultiWriter:
    def __init__(self, filename):
        self.log = open(filename, 'w', encoding='utf-8')
        self.out = Output()  # Output widget to display output in the notebook
        display(self.out)    # Display the widget in the notebook cell

    def write(self, message):
        # Write to the file
        self.log.write(message)
        self.log.flush()

        # Write to the notebook cell output without recursion
        with self.out:
            self.out.append_stdout(message)  # Avoids calling print()

    def flush(self):
        self.log.flush()

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

if __name__ == "__main__":
    # First ensure directories exist
    if not setup_directories():
        print("Failed to create necessary directories. Exiting.")
        sys.exit(1)

    # set_all_seeds(RANDOM_SEED, gpu_deterministic=False)

    timestamp = datetime.now().strftime('%Y-%m-%d-%Hh-%Mm-%Ss')
    filename = os.path.join(RESULTS_DIR, f"{EXP_ID}-dir_{experiment_parameters['data_dir'].split('/')[-2]}_{experiment_parameters['data_dir'].split('/')[-1]}-mode_{experiment_parameters['mode']}-split_{experiment_parameters['split_index']}-arch_{experiment_parameters['architecture']}-clf_{experiment_parameters['classifier']}.txt")

    print(f"> Output will be saved to: {filename}")

    try:
        sys.stdout = MultiWriter(filename)

        check_environment()

        print("=== Starting Experiment ===")
        display_configurations()

        start_time = datetime.now()
        main()
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
        print("=== PROCESS FINISHED ===")