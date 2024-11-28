import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import GlobalAveragePooling2D
from tabulate import tabulate
import sys
from datetime import datetime
import tensorflow as tf
import numpy as np

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f'exp-06/model_summaries/modsum_{timestamp}.txt'

hyperparameters = {
    # Data loading and splitting
    'img_size': (128, 128),
    'color_mode': 'grayscale',
    'test_split_size': 0.2,
    'test_split_index': 0,

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
    'lr_reduction_factor': 0.5,

    # Downstream task
    'feature_extraction_layer': -2,
    'rf_n_estimators': 100,
    'svm_kernel': 'linear',
    'gb_n_estimators': 100,
    'xgb_n_estimators': 100,
    'cv_splits': 5,
}

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

def build_custom_cnn_model(input_shape=(128, 128, 1), num_classes=4, architecture_name='cnn', fine_tune=True):
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

    return model

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
                print(f"Error: Layer '{layer_name}' not found. Available layers:")
                for idx, layer in enumerate(model.layers):
                    print(f"{idx}: {layer.name}")
                raise e
        
        # Get outputs
        outputs = {}
        for layer_name, inspection_model in inspection_models.items():
            outputs[layer_name] = inspection_model.predict(input_data)
            
        return outputs

    return inspect_layer

def example_layer_inspection(architecture_name='cnn'):
    """
    Example of how to use the layer inspector with different architectures
    
    Args:
        architecture_name: The architecture to inspect ('cnn', 'efficientnetb0', etc.)
    """
    # Create model
    model = build_custom_cnn_model(architecture_name=architecture_name)
    
    # Create inspector
    inspector = create_layer_inspector(model)
    
    # Generate sample data
    sample_data = np.random.random((1, 128, 128, 1))
    
    # Get all layer names
    print("\nAvailable layers:")
    for idx, layer in enumerate(model.layers):
        print(f"{idx}: {layer.name}")
    
    # Define layers to inspect based on architecture
    if architecture_name == 'cnn':
        layers_to_inspect = [
            'conv2d',           # First convolution
            'batch_normalization',  # First batch norm
            'max_pooling2d',    # First pooling
            'gap_cnn',          # Global pooling
            'predictions'       # Final layer
        ]
    else:  # Transfer learning architectures
        layers_to_inspect = [
            'resizing',
            'channel_expansion',
            'gap_transfer',
            'predictions'
        ]
    
    print(f"\nInspecting layers for {architecture_name} architecture:")
    print(f"Attempting to inspect: {layers_to_inspect}")
    
    # Inspect layers that exist in the model
    existing_layers = [layer.name for layer in model.layers]
    valid_layers = [layer for layer in layers_to_inspect if layer in existing_layers]
    
    if not valid_layers:
        print("No specified layers found in model. Available layers are:")
        for layer_name in existing_layers:
            print(f"- {layer_name}")
        return
    
    try:
        outputs = inspector(sample_data, valid_layers)
        
        # Print output information
        for layer_name, output in outputs.items():
            print(f"\n{layer_name} output:")
            print(f"Shape: {output.shape}")
            print(f"Range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Mean: {output.mean():.3f}")
            print(f"Std: {output.std():.3f}")
            
            # Additional analysis for specific layer types
            if 'conv' in layer_name.lower():
                print("Channel statistics:")
                for i in range(output.shape[-1]):
                    channel_data = output[..., i]
                    print(f"Channel {i}: Mean={channel_data.mean():.3f}, Std={channel_data.std():.3f}")
            
            elif 'batch_normalization' in layer_name.lower():
                print("Batch normalization stats (should be roughly mean=0, std=1):")
                print(f"Global Mean: {output.mean():.3f}")
                print(f"Global Std: {output.std():.3f}")
    
    except Exception as e:
        print(f"Inspection failed: {str(e)}")

def inspect_intermediate_activations(model, input_data, layer_names=None):
    """
    More comprehensive inspection of intermediate layer activations
    
    Args:
        model: Keras model to inspect
        input_data: Input data to run through the model
        layer_names: Optional list of layer names to inspect. If None, inspects all layers
    """
    if layer_names is None:
        layer_names = [layer.name for layer in model.layers]
    
    inspector = create_layer_inspector(model)
    
    try:
        outputs = inspector(input_data, layer_names)
        
        print("\n=== Activation Analysis ===")
        for layer_name, output in outputs.items():
            print(f"\nLayer: {layer_name}")
            print("-" * 50)
            
            # Basic statistics
            print("Basic Statistics:")
            print(f"Shape: {output.shape}")
            print(f"Total elements: {output.size:,}")
            print(f"Range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"Mean: {output.mean():.3f}")
            print(f"Std: {output.std():.3f}")
            
            # Activation pattern analysis
            zeros = np.sum(np.abs(output) < 1e-6)
            zero_percent = zeros / output.size * 100
            print(f"\nActivation Patterns:")
            print(f"Zero activations: {zeros:,} ({zero_percent:.1f}%)")
            
            # Distribution analysis
            percentiles = np.percentile(output, [25, 50, 75])
            print(f"\nDistribution:")
            print(f"25th percentile: {percentiles[0]:.3f}")
            print(f"Median: {percentiles[1]:.3f}")
            print(f"75th percentile: {percentiles[2]:.3f}")
            
            # Layer-specific analysis
            layer = model.get_layer(layer_name)
            if isinstance(layer, tf.keras.layers.Conv2D):
                print("\nConvolution Layer Analysis:")
                print(f"Number of filters: {output.shape[-1]}")
                print(f"Feature map size: {output.shape[1]}x{output.shape[2]}")
                
                # Analyze each filter's activation
                filter_means = np.mean(output, axis=(0,1,2))
                active_filters = np.sum(filter_means > 0.1)
                print(f"Active filters (mean > 0.1): {active_filters} out of {output.shape[-1]}")
                
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                print("\nBatch Normalization Analysis:")
                print("Checking if normalization is working properly...")
                is_normalized = abs(output.mean()) < 0.1 and abs(output.std() - 1) < 0.5
                print(f"Properly normalized: {is_normalized}")
                
            elif isinstance(layer, tf.keras.layers.Dense):
                print("\nDense Layer Analysis:")
                neuron_means = np.mean(output, axis=0)
                active_neurons = np.sum(neuron_means > 0.1)
                print(f"Active neurons (mean > 0.1): {active_neurons} out of {output.shape[-1]}")
    
    except Exception as e:
        print(f"Detailed inspection failed: {str(e)}")

def test_all_models_with_summary():
    """
    Test all available model architectures and save their summaries
    """
    architectures = [
        'cnn',
        'unet',
        'resnet50',
        'resnet101',
        'resnet152',
        'efficientnetb0',
        'vgg16',
        'vgg19',
        'inceptionv3'
    ]
    
    output_file = OUTPUT_FILE
    
    configs = [
        {'input_shape': (128, 128, 1), 'num_classes': 4},
        # {'input_shape': (256, 256, 1), 'num_classes': 4},
    ]
    
    # Redirect stdout to file
    original_stdout = sys.stdout
    with open(output_file, 'w') as f:
        sys.stdout = f
        
        print("=" * 80)
        print("MODEL ARCHITECTURE SUMMARIES")
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        for arch in architectures:
            print("=" * 80)
            print(f"ARCHITECTURE: {arch.upper()}")
            print("=" * 80)
            
            for config in configs:
                print("-" * 80)
                print(f"Configuration: Input Shape={config['input_shape']}, "
                     f"Classes={config['num_classes']}")
                print("-" * 80)
                
                try:
                    model = build_custom_cnn_model(
                        input_shape=config['input_shape'],
                        num_classes=config['num_classes'],
                        architecture_name=arch
                    )
                    
                    # Only print summary once per model configuration
                    print_model_summary(model, method='compact')
                    
                    # Clean up
                    del model
                    tf.keras.backend.clear_session()
                    
                except Exception as e:
                    print(f"Error creating model: {str(e)}\n")
        
        sys.stdout = original_stdout
    
    print(f"Model summaries have been saved to: {output_file}")
    return output_file

def check_models():
    """
    Main function to run model tests and generate summaries
    """
    try:
        print("Starting model architecture tests...")
        
        # Test all models and generate summaries using print_model_summary
        summary_file = test_all_models_with_summary()
         
        print("\nAll tests completed successfully!")
        print(f"Detailed summaries have been saved to: {summary_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0
    
def check_layers():
    # Test with CNN architecture
    print("\nTesting CNN architecture:")
    example_layer_inspection('cnn')
    
    # Test with transfer learning architecture
    print("\nTesting EfficientNetB0 architecture:")
    example_layer_inspection('efficientnetb0')
    
    # Detailed activation analysis
    model = build_custom_cnn_model(architecture_name='cnn')
    sample_data = np.random.random((1, 128, 128, 1))
    inspect_intermediate_activations(model, sample_data, ['conv2d', 'batch_normalization', 'predictions'])

if __name__ == "__main__":
    # sys.exit(check_models())
    sys.exit(check_layers())