# AISystem-2402

This repository provides implementations of both few-shot and zero-shot anomaly detection using CLIP embeddings.

## Project Structure
```
AISystem-2402
├── fewshot-clip/
├── zeroshot-clip/
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset Structure

**Training Dataset**
```
└──── ./train
    ├──── {class_name}/
    |   ├──── 0000.jpg
    |   ├──── ...
    |   └──── 0000.jpg
    ├──── {class_name}/
    |   ├──── 0000.jpg
    |   ├──── ...
    |   └──── 0000.jpg
```

**Test (Validation) Dataset**
```
└──── ./test
    ├──── anomaly/
    |   ├──── 0000.jpg
    |   ├──── ...
    |   └──── 0000.jpg
    ├──── normal/
    |   ├──── 0000.jpg
    |   ├──── ...
    |   └──── 0000.jpg
    
```

The dataset for anomaly detection is organized to facilitate both few-shot and zero-shot approaches with CLIP embeddings.

### Dataset Download
 Dataset can be downloaded [here](https://drive.google.com/file/d/1jOXzkmMIkYCCvC50J26V6mXn7wEYYpxP/view?usp=drive_link).

### Dataset Organization
**Train Dataset**: This dataset provides 5 `normal` images per each of the 16 classes. You may use or manipulate this dataset to apply few-shot or zero-shot strategies.

**Validation Dataset**: Referred to as the "test" dataset in the code, the validation dataset contains both `noraml` and `anomaly` images. It can be used to evaluate model performance during development.

**Test Dataset**: Since test dataset is private, there is **no** separate test dataset provided for this project. 

> Each class within the anomaly dataset contains at least one example of an anomaly, such as dots, cuts, or other class-specific defects like below. This design encourages exploration of anomaly detection within constrained data conditions.


<p align="center">
  <img src="docs/0001.jpg" width="45%" alt="Image 1">
  <img src="docs/0002.jpg" width="45%" alt="Image 2">
</p>
## Getting Started

1. Clone the Repository
```
git clone https://github.com/PiLab-CAU/AISystem-2402.git
```

2. Set up the Conda Environment
```
conda create -n cauclip python=3.xx
```
> The Recommended Python version is 3.8 or above :wink:

```
conda activate cauclip
```

3. Navigate to the Cloned Folder
```
cd AISystem-2402
```

4. Install Dependencies
```
pip install -r requirements.txt
```

Once the dependencies are installed, we are ready to detect anomalies! :grinning:

## Run Sample Code
### Fewshot-CLIP
To run the sample code for Few-Shot CLIP Embedding, execute:

```
python fewshot-clip/main.py
```

### Zeroshot-CLIP
To run the sample code for Zero-Shot CLIP Embedding, execute:

```
python zeroshot-clip/main.py
```

## Performance Summary
The Performance of sample codes is shown in the tables below.


## Few-Shot CLIP

| Metric                    | Value                  |
|---------------------------|------------------------|
| **Train Loss (Epoch 1)**  | 0.7279                 |
| **Train Accuracy (Epoch 1)** | 52.86%            |
| **Test Accuracy (Epoch 1)**  | 63.08%            |
| **Best Model Path**       | `saved_models/best_model.pth` (Accuracy: 63.08%) |
| **Overall Test Accuracy** | 32.31%                |
| **Total Time Taken**      | 18.42 seconds         |
| **Average Time per Image** | 0.2834 seconds       |

#### Inference Details
| Class   | Correct | Total | Accuracy | Normal Similarity | Anomaly Similarity |
|---------|---------|-------|----------|-------------------|--------------------|
| Normal  | 21      | 21    | 100.00%  | -                | -                 |
| Anomaly | 0       | 44    | 0.00%    | -                | -                 |

---

## Zero-Shot CLIP

| Metric                       | Value                  |
|------------------------------|------------------------|
| **Total Images**             | 65                     |
| **Correct Predictions**      | 35                     |
| **Overall Accuracy**         | 53.85%                 |

#### Class-Specific Performance

| Class     | Total | Correct | Incorrect | Accuracy | Avg Anomaly Score | Avg Normal Similarity | Avg Anomaly Similarity |
|-----------|-------|---------|-----------|----------|-------------------|-----------------------|------------------------|
| **Normal** | 21    | 21      | 0         | 100.00%  | 0.288            | 0.921                 | 0.634                  |
| **Anomaly** | 44    | 14      | 30        | 31.82%   | 0.234            | 0.862                 | 0.627                  |

#### Detailed Metrics

| Metric      | Value     |
|-------------|-----------|
| **Precision** | 100.00% |
| **Recall**    | 31.82%  |
| **F1 Score**  | 48.28%  |

*Metrics saved in `./results/metrics_{datetime}_{time}.json`. Results can be found in the `./results` directory.*

## Further Information
For additional details on each module, check out the specific README files:

- [Few-Shot CLIP](./fewshot-clip/README.md) 

- [Zero-Shot CLIP](./zeroshot-clip/README.md)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-lightblue.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contact

For questions or further information, please contact [leessoit@gmail.com](mailto:leessoit@gmail.com) or use the [issue tab](https://github.com/PiLab-CAU/AISystem-2402/issues) to report any problems or suggestions.
