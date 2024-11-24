# ZeroShot-Clip

This project implements an anomaly detection system using a CLIP-based classifier. The model is trained to classify images as either "normal" or "anomaly" based on their visual features.

## Project Structure
```
AISystem-2402/zeroshot-clip
├── models
│   ├── anomaly_detector.py
│   └── clip_model.py
├── utils
│   ├── augmentation
│   │   ├── anomaly_augmenter.py
│   │   ├── base.py
│   │   ├── color.py
│   │   ├── geometric.py
│   │   └── noise.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── visualization.py
├── config.py
└── main.py
```


- **`data_loader.py`**: Defines the `CustomDataset` class for loading and preprocessing images, specifically for the anomaly detection task. It includes augmentation methods to create anomaly samples by applying transformations like random masking and image alterations.

- **`model.py`**: Defines the `CLIPClassifier` model, which utilizes the CLIP model's image encoder as a feature extractor, followed by a fully connected layer for binary classification between "normal" and "anomaly." It also provides a function to load the pre-trained CLIP model for classification.

- **`main.py`**: The main script for training the model. It initializes data loaders, trains the model using binary cross-entropy loss, evaluates accuracy, and saves the best-performing model to the `saved_models/` directory.

- **`inference.py`**: Script to perform inference on a test dataset. It loads the trained model, makes predictions for images in the test directory, and saves labeled images with prediction results. It also outputs accuracy metrics.

- **`utils/`**:
  - **`augmentations.py`**: Provides functions to apply random augmentations to images, including color adjustments, rotation, flipping, Gaussian blurring, noise addition, and random masking to simulate anomalies. It contains:
    
  - **`visualize.py`**: Contains visualization utilities, specifically `save_prediction_image`, which creates and saves an image displaying the original input alongside the model's label and prediction results. It sets up a two-panel display, with the image on one side and text on the other, and saves the visualization in the specified output directory.

