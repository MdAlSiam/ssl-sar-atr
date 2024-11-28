import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import bm3d
import random
from enum import Enum
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend
from PIL import Image  # Added PIL for more robust image loading
import matplotlib.pyplot as plt

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
    
    if pretext_task == PreextTask.ORIGINAL:
        return aug_image, PreextTask.ORIGINAL.value
        
    elif pretext_task == PreextTask.ROTATE_90:
        rows, cols = aug_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        aug_image = cv2.warpAffine(aug_image, M, (cols, rows))
        return aug_image, PreextTask.ROTATE_90.value
        
    elif pretext_task == PreextTask.ROTATE_180:
        rows, cols = aug_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
        aug_image = cv2.warpAffine(aug_image, M, (cols, rows))
        return aug_image, PreextTask.ROTATE_180.value
        
    elif pretext_task == PreextTask.ROTATE_270:
        rows, cols = aug_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
        aug_image = cv2.warpAffine(aug_image, M, (cols, rows))
        return aug_image, PreextTask.ROTATE_270.value
        
    elif pretext_task == PreextTask.BLUR:
        sigma = random.uniform(0.5, 1.0)
        aug_image = gaussian_filter(aug_image, sigma=sigma)
        return aug_image, PreextTask.BLUR.value
        
    elif pretext_task == PreextTask.FLIP_HORIZONTAL:
        aug_image = cv2.flip(aug_image, 1)
        return aug_image, PreextTask.FLIP_HORIZONTAL.value
        
    elif pretext_task == PreextTask.FLIP_VERTICAL:
        aug_image = cv2.flip(aug_image, 0)
        return aug_image, PreextTask.FLIP_VERTICAL.value
        
    elif pretext_task == PreextTask.FLIP_BOTH:
        aug_image = cv2.flip(aug_image, -1)
        return aug_image, PreextTask.FLIP_BOTH.value
        
    elif pretext_task == PreextTask.DENOISE:
        sigma_psd = 25/255
        aug_image = bm3d.bm3d(aug_image, sigma_psd=sigma_psd)
        return aug_image, PreextTask.DENOISE.value
    
    elif pretext_task == PreextTask.ZOOM_IN:
        scale_factor = random.uniform(1.2, 1.5)  # Zoom in 20-50%
        aug_image = zoom_image(aug_image, scale_factor)
        return aug_image, PreextTask.ZOOM_IN.value
        
    elif pretext_task == PreextTask.ZOOM_OUT:
        scale_factor = random.uniform(0.6, 0.8)  # Zoom out 20-40%
        aug_image = zoom_image(aug_image, scale_factor)
        return aug_image, PreextTask.ZOOM_OUT.value

def preprocess_data_pretext(images):
    """
    Preprocess images for pretext tasks in representation learning
    """
    augmented_images = []
    pretext_labels = []
    original_indices = []
    
    for idx, image in enumerate(images):
        for task in PreextTask:
            aug_image, task_label = augment_single_image(image, task)
            augmented_images.append(aug_image)
            pretext_labels.append(task_label)
            original_indices.append(idx)
    
    augmented_images = np.array(augmented_images)
    pretext_labels = np.array(pretext_labels)
    original_indices = np.array(original_indices)
    
    print(f'> Generated {len(augmented_images)} images of shape {augmented_images[0].shape}')
    print(f'> Number of pretext tasks (classes): {len(PreextTask)}')
    print(f'> Pretext task labels: {sorted(set(pretext_labels))}')
    
    return augmented_images, pretext_labels, original_indices

# Create a more visually interesting dummy image
def create_dummy_image(size=64):
    # Create a structured pattern instead of random noise
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Create RGB channels with different patterns
    r = np.sin(xx * 10) * 0.5 + 0.5
    g = np.cos(yy * 10) * 0.5 + 0.5
    b = np.sin((xx + yy) * 5) * 0.5 + 0.5
    
    # Combine channels
    image = np.stack([r, g, b], axis=-1)
    return image

def load_image(image_path, size=None):
    """
    Load and preprocess an image for augmentation
    
    Parameters:
        image_path: Path to the image file
        size: Optional tuple of (height, width) to resize the image
    
    Returns:
        image: Normalized numpy array in range [0, 1]
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    # Load image using PIL (supports more formats than OpenCV)
    image = Image.open(image_path)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if specified
    if size is not None:
        image = image.resize((size, size), Image.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    image = np.array(image).astype(np.float32) / 255.0
    
    print(f"Loaded image shape: {image.shape}")
    print(f"Image value range: [{image.min():.3f}, {image.max():.3f}]")
    
    return image

def visualize_augmentations(original_image, augmented_images, pretext_labels, output_dir='./outputs'):
    """Visualize original image against all its augmentations"""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'augmentations-5.png')
    
    # Adjust figure size for additional augmentations
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))  # Changed to 3x4 grid
    fig.suptitle('Original Image vs Augmentations', fontsize=14)
    
    axes = axes.flatten()
    
    original_image_normalized = np.clip(original_image, 0, 1)
    axes[0].imshow(original_image_normalized)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for i, (aug_image, label) in enumerate(zip(augmented_images, pretext_labels)):
        if i >= len(axes):  # Skip if we run out of subplot space
            break
        ax = axes[i]
        aug_image_normalized = np.clip(aug_image, 0, 1)
        ax.imshow(aug_image_normalized)
        ax.set_title(f'{PreextTask(label).name}')
        ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\nVisualization saved to: {save_path}")
    return save_path

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create output directory
    output_dir = './pretext_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create one dummy image with a pattern
    dummy_image = load_image('/mnt/d/SAMPLE_dataset_public/png_images/qpm/real/bmp2/bmp2_real_A_elevDeg_016_azCenter_037_49_serial_9563.png')
    dummy_images = [dummy_image]
    
    # Generate augmentations
    aug_images, pretext_labels, orig_indices = preprocess_data_pretext(dummy_images)
    
    # Get augmentations for the first image only
    first_image_mask = (orig_indices == 0)
    first_image_augs = aug_images[first_image_mask]
    first_image_labels = pretext_labels[first_image_mask]
    
    # Visualize and save
    output_path = visualize_augmentations(
        dummy_image, 
        first_image_augs, 
        first_image_labels,
        output_dir=output_dir
    )
    
    # Print distribution of tasks
    print("\nTask Distribution:")
    unique_labels, counts = np.unique(pretext_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Task: {PreextTask(label).name}, Count: {count}")