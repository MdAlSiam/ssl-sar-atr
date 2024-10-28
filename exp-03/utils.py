import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras import layers, models, regularizers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

from config import *
from data_processor import *

#====================== PRETEXT ====================#

# Data Augmentation with diverse transformations (Rotation, Brightness, Scaling, Translation, etc.)
def augment_image(image, rotation_angle):
    """Apply a diverse set of augmentations to the image."""
    '''
    image = tf.image.random_flip_left_right(image)  # Flip horizontally
    image = tf.image.random_brightness(image, max_delta=0.2)  # Brightness adjustment
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Contrast adjustment
    image = tf.image.random_zoom(image, (0.8, 1.2))  # Zooming simulation
    image = tf.image.random_translation(image, translations=[5, 5])  # Random Translation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # Saturation adjustment
    image = tf.image.random_hue(image, max_delta=0.2)  # Hue adjustment
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=50, max_jpeg_quality=100)  # JPEG quality jitter
    image = tf.image.resize_with_crop_or_pad(image, 70, 70)  # Zooming simulation via resizing
    image = tf.image.random_crop(image, size=[64, 64, 1])  # Crop back to original size
    '''
    # Apply rotation (keeping the original logic)
    # rotation_angle = np.random.choice([0, 90, 180, 270])
    if rotation_angle == 90:
        image = tf.image.rot90(image)
    elif rotation_angle == 180:
        image = tf.image.rot90(image, k=2)
    elif rotation_angle == 270:
        image = tf.image.rot90(image, k=3)

    label = rotation_angle // 90
    return image, label

def preprocess_data(images):
    augmented_images = []
    labels = []
    for image in images:
        for rotation_angle in [0, 90, 180, 270]:
            aug_image, label = augment_image(image, rotation_angle)
            augmented_images.append(aug_image)
            labels.append(label)
    print(f'> {len(augmented_images)} augmented images generated each of shape {augmented_images[0].shape} with {len(labels)} labels')
    return np.array(augmented_images), np.array(labels)

# Balanced Sampling using Oversampling (Copies data from minor class to balance number of samples)
def balance_classes(x_data, y_data):
    ros = RandomOverSampler(random_state=RANDOM_SEED)
    x_flattened = x_data.reshape((x_data.shape[0], -1))  # Reshape to 2D for balancing
    x_resampled, y_resampled = ros.fit_resample(x_flattened, y_data)
    x_resampled = x_resampled.reshape((-1, x_data.shape[1], x_data.shape[2], x_data.shape[3]))  # Reshape back to original
    print(f'> (balancing) resampled to {x_resampled.shape[0]} samples')
    return x_resampled, y_resampled

# Helper function to build U-Net (custom   implementation)
def build_unet_model(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoding (down-sampling) path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bridge (bottom of the U-Net)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)

    # Decoding (up-sampling) path
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.Concatenate()([u1, c2])

    u2 = layers.UpSampling2D((2, 2))(u1)
    u2 = layers.Concatenate()([u2, c1])

    outputs = layers.Conv2D(3, (1, 1), activation='softmax')(u2)

    return models.Model(inputs=inputs, outputs=outputs)

def build_custom_cnn_model(input_shape=(128, 128, 1), num_classes=4, architecture_name=None, fine_tune=True):
    print(f'> pretext task training model: input_shape {input_shape}, num_classes {num_classes}, architecture_name {architecture_name}')

    inputs = tf.keras.Input(shape=input_shape)

    if architecture_name != 'cnn':
        # Upsample input to the required size if needed (InceptionV3 requires minimum 75x75)
        if architecture_name == 'inceptionv3' and (input_shape[0] < 75 or input_shape[1] < 75):
            required_size = (75, 75)  # InceptionV3 minimum size
        else:
            required_size = (input_shape[0], input_shape[1])  # Default size for other models

        x = layers.Resizing(required_size[0], required_size[1])(inputs)  # Resize input to required size

        # Convert grayscale (1-channel) to 3-channel RGB for pretrained models
        x = layers.Conv2D(3, (1, 1))(x)

        # Pretrained model selection logic
        # ResNet Variants
        if architecture_name == 'resnet50':
            from tensorflow.keras.applications import ResNet50
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(required_size[0], required_size[1], 3))
        elif architecture_name == 'resnet101':
            from tensorflow.keras.applications import ResNet101
            base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(required_size[0], required_size[1], 3))
        elif architecture_name == 'resnet152':
            from tensorflow.keras.applications import ResNet152
            base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(required_size[0], required_size[1], 3))

        # EfficientNetB0
        elif architecture_name == 'efficientnetb0':
            from tensorflow.keras.applications import EfficientNetB0
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(required_size[0], required_size[1], 3))

        # VGGNet Variants
        elif architecture_name == 'vgg16':
            from tensorflow.keras.applications import VGG16
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(required_size[0], required_size[1], 3))
        elif architecture_name == 'vgg19':
            from tensorflow.keras.applications import VGG19
            base_model = VGG19(weights='imagenet', include_top=False, input_shape=(required_size[0], required_size[1], 3))

        # InceptionV3
        elif architecture_name == 'inceptionv3':
            from tensorflow.keras.applications import InceptionV3
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(required_size[0], required_size[1], 3))

        # U-Net (not from Keras applications, custom U-Net function)
        elif architecture_name == 'unet':
            base_model = build_unet_model(input_shape=(required_size[0], required_size[1], 3))  # Custom function to build U-Net model

        # Set base model to non-trainable if fine-tuning is disabled
        if not fine_tune:
            base_model.trainable = False
        else:
            base_model.trainable = True

        # Apply base model to input
        x = base_model(x)
        x = GlobalAveragePooling2D()(x)
    else:
        # Build a custom CNN if no pretrained model is specified
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary(line_length=120, expand_nested=True)

    return model

# Step 7: Plot Results
def plot_results(history):
    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()

    plt.show()
    plt.close()
    print()

# Function to save the trained model
def save_model(model, architecture_name, data_dir, split_index):
    model_name = f"pretext_model_{data_dir.split('/')[-2]}_{data_dir.split('/')[-1]}_{architecture_name}_split_{split_index}.h5"
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)

    # Check if the file exists after saving
    if os.path.isfile(model_path):
        print(f"> Model successfully saved as {model_path}")
    else:
        print("> ERROR: Model file not found after saving.")


# Updated pretext training pipeline
def run_pretext_pipeline(data_path, architecture_name, split_index):
    """Train and save pretext model using saved processed data"""
    # Load processed data
    x_train, y_train, _, _ = load_processed_data(data_path)
    
    # Preprocess data for pretext task
    x_augmented, y_augmented = preprocess_data(x_train)
    
    # Build and train model
    model = build_custom_cnn_model(
        input_shape=hyperparameters['input_shape'],
        num_classes=hyperparameters['num_classes'],
        architecture_name=architecture_name,
        fine_tune=hyperparameters['fine_tune']
    )
    
    # Configure callbacks
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
        )
    ]
    
    # Train model
    history = model.fit(
        x_augmented, y_augmented,
        batch_size=hyperparameters['batch_size'],
        epochs=hyperparameters['epochs'],
        validation_split=hyperparameters['validation_split'],
        callbacks=callbacks
    )
    
    # Save model
    save_model(model, architecture_name, data_path, split_index)
    
    return history

#====================== DOWNSTREAM ====================#

def load_model(architecture_name, data_dir, split_index):
    model_name = f"pretext_model_{data_dir.split('/')[-2]}_{data_dir.split('/')[-1]}_{architecture_name}_split_{split_index}.h5"
    model_path = os.path.join(MODEL_DIR, model_name)

    # Check if the file exists before loading
    if os.path.isfile(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model successfully loaded from {model_path}")

        # Sanity check: test with a small batch of random data
        random_input = np.random.rand(1, 128, 128, 1)  # Adjust the input shape as per your model requirements
        try:
            prediction = model.predict(random_input)
            # print("Sanity check passed: Model loaded and successfully made predictions.")
            return model
        except Exception as e:
            print(f"Sanity check failed: {e}")
            return None
    else:
        print("Error: Model file not found. Load operation failed.")
        return None

# Feature Extraction from deeper layers
def extract_features(pretext_model, x_data, layer_index=-2):
    intermediate_model = models.Model(inputs=pretext_model.input, outputs=pretext_model.layers[layer_index].output)
    # print('> intermediate feature extractor model:')
    # intermediate_model.summary()
    features = intermediate_model.predict(x_data)
    print(f'> extracted features of shape {features.shape}')
    return features

def evaluate_downstream_task(clf, X_test, y_test):
    """Evaluate the downstream task on the held-out test set"""
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def train_downstream_task(train_features, train_labels, test_features, test_labels, classifier='random_forest', n_splits=5):
    """Modified to include validation during training and final evaluation on test set"""
    print(f'> performing downstream task with {classifier}')

    if classifier == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    elif classifier == 'svm':
        clf = SVC(kernel='linear', random_state=RANDOM_SEED)
    elif classifier == 'gradient_boosting':
        clf = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
    elif classifier == 'xgboost':
        clf = XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, use_label_encoder=True)

    # First, perform cross-validation on training data
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = {
        'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': []
    }

    # print("\nCross-validation results:")
    for train_idx, val_idx in skf.split(train_features, train_labels):
        X_train_fold, X_val_fold = train_features[train_idx], train_features[val_idx]
        y_train_fold, y_val_fold = train_labels[train_idx], train_labels[val_idx]

        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_val_fold)

        cv_scores['accuracies'].append(accuracy_score(y_val_fold, y_pred))
        cv_scores['precisions'].append(precision_score(y_val_fold, y_pred, average='macro'))
        cv_scores['recalls'].append(recall_score(y_val_fold, y_pred, average='macro'))
        cv_scores['f1_scores'].append(f1_score(y_val_fold, y_pred, average='macro'))

    # Train final model on full training data
    clf.fit(train_features, train_labels)

    # Evaluate on test set
    test_metrics = evaluate_downstream_task(clf, test_features, test_labels)

    return clf, cv_scores, test_metrics

# Updated downstream pipeline
def run_downstream_pipeline(data_path, architecture_name, downstream_classifier, split_index):
    """Run downstream task using saved data and pretext model"""
    # Load processed data
    x_train, y_train, x_test, y_test = load_processed_data(data_path)
    
    # Load pretext model
    pretext_model = load_model(architecture_name, data_path, split_index)
    
    # Extract features
    train_features = extract_features(pretext_model, x_train, hyperparameters['feature_extraction_layer'])
    test_features = extract_features(pretext_model, x_test, hyperparameters['feature_extraction_layer'])
    
    # Train and evaluate downstream task
    clf, cv_scores, test_metrics = train_downstream_task(
        train_features, y_train,
        test_features, y_test,
        classifier=downstream_classifier,
        n_splits=hyperparameters['cv_splits']
    )
    
    return clf, cv_scores, test_metrics

