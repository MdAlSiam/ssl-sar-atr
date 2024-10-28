import os

RANDOM_SEED = 42
MODEL_DIR = os.path.join("exp-02", "pretext_models")

data_dirs = [
    '/mnt/c/Users/Siam/OneDrive - Tuskegee University/ai-arni-nsf/SAMPLE_dataset_public/png_images/qpm/synth',
    '/mnt/c/Users/Siam/OneDrive - Tuskegee University/ai-arni-nsf/SAMPLE_dataset_public/png_images/decibel/real',
    '/mnt/c/Users/Siam/OneDrive - Tuskegee University/ai-arni-nsf/SAMPLE_dataset_public/png_images/decibel/synth'
]

architecture_names = [
    'cnn',
    'resnet50',
    'resnet101',
    'resnet152',
    'efficientnetb0',
    'vgg16',
    'vgg19',
    'inceptionv3',
    'unet'
]

classifiers = ['random_forest', 'svm', 'gradient_boosting', 'xgboost']

hyperparameters = {
    'hyperparameter': 0
}