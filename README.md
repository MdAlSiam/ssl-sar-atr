# Self-Supervised Learning on SAMPLE Dataset
A deep learning implementation for self-supervised learning using rotation prediction as a pretext task on the SAMPLE dataset, followed by downstream classification tasks.

## Project Overview
This project implements a self-supervised learning pipeline that:
1. Uses rotation prediction (0°, 90°, 180°, 270°) as a pretext task
2. Supports multiple deep learning architectures (CNN, ResNet, VGG, etc.)
3. Implements various downstream classifiers (Random Forest, SVM, Gradient Boosting, XGBoost)
4. Handles both real and synthetic data from the SAMPLE dataset

## Features
- Multiple architecture support (CNN, ResNet50/101/152, EfficientNetB0, VGG16/19, InceptionV3, U-Net)
- Data augmentation and balancing
- Feature extraction from intermediate layers
- Multiple downstream classifiers
- Cross-validation for robust evaluation
- Performance visualization
- Model saving and loading capabilities

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Setup
1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Organize your SAMPLE dataset in the following structure:
```
SAMPLE_dataset_public/
├── png_images/
│   ├── qpm/
│   │   ├── real/
│   │   └── synth/
│   └── decibel/
│       ├── real/
│       └── synth/
```

2. Update the `data_dirs` list in the code with your dataset paths.

3. Run the pretext task training:
```python
for data_dir in data_dirs:
    for architecture_name in architecture_names:
        run_pretext_pipeline(data_dir, architecture_name)
```

4. Run the downstream task evaluation:
```python
for data_dir in data_dirs:
    for architecture_name in architecture_names:
        for classifier in classifiers:
            run_downstream_pipeline(data_dir, architecture_name, classifier, -2)
```

## Configuration
- Modify `RANDOM_SEED` for reproducibility
- Adjust image size in `load_data_from_directory()`
- Configure model architectures in `architecture_names`
- Modify classifier options in `classifiers`
- Adjust training parameters in `build_custom_cnn_model()`

## Results
The code generates:
- Training/validation loss and accuracy plots
- Performance metrics for downstream tasks:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## File Structure
```
.
├── README.md
├── requirements.txt
├── exp-01.py
└── exp-01.ipynb
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details

## Contact
Md Al Siam - md.al.siam.008@gmail.com
Project Link: [https://github.com/MdAlSiam/ssl-sar-atr](https://github.com/MdAlSiam/ssl-sar-atr)

## Acknowledgments
- SAMPLE dataset creators and contributors
- TensorFlow team for the deep learning framework
- scikit-learn team for the machine learning tools

---
