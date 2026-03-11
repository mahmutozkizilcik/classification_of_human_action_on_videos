# Human Activity Recognition using Skeleton-Based Features

**Mahmut Özkızılcık - 2220765019**

## 📌 Project Overview

This project implements a **video-based Human Activity Recognition (HAR)** system that classifies six human actions: **boxing, handclapping, handwaving, jogging, running, and walking**.

The pipeline consists of two main stages:
1. **Feature Extraction** — Using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) (BODY_25 model) to extract 2D skeleton keypoints from raw video frames
2. **Classification** — Four different machine learning / deep learning models are trained and compared

## 🏗️ Project Structure

```
├── FeatureExtraction.py        # OpenPose-based skeleton keypoint extraction
├── FeatureExtraction.ipynb     # Notebook version of feature extraction
├── project.py                  # All model implementations (GAK+SVM, Shapelet+MLP, LSTM, 1D-CNN)
├── project.ipynb               # Notebook version with outputs and visualizations
├── extracted_features/         # Extracted .npz files (download link below)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔬 Methodology

### Feature Extraction (OpenPose)
- Each video frame is processed to detect **25 body keypoints** using the BODY_25 model
- Coordinates are **normalized** relative to video resolution (160×120): `x/160`, `y/120`
- Results are saved as compressed `.npz` files containing time-series skeleton data

### Models Implemented

| # | Model | Description | Best Accuracy |
|---|-------|-------------|:------------:|
| 1 | **GAK + SVM** | Global Alignment Kernel with Support Vector Machine | ~82.00% |
| 2 | **Shapelet + MLP** | Shapelet Transform followed by a PyTorch MLP | 45.83% |
| 3 | **LSTM** | Long Short-Term Memory sequence classifier (PyTorch) | 86.67% |
| 4 | **1D-CNN** | 1D Convolutional Neural Network (PyTorch) | **90.00%** |

### Performance Leaderboard

| Rank | Model | Key Strength |
|:----:|-------|-------------|
| 🥇 | **1D-CNN** | Best overall generalization and speed differentiation |
| 🥈 | **LSTM** | Strong on upper-body actions, captured long-term dependencies |
| 🥉 | **GAK + SVM** | Perfect separation of upper vs. lower body groups |
| 4 | **Shapelet + MLP** | Limited capacity for complex temporal patterns |

## 📊 Key Findings

- **Deep Learning models** (1D-CNN, LSTM) significantly outperformed traditional approaches
- **1D-CNN** achieved the best accuracy (90%) — convolutional filters effectively captured local temporal patterns
- The most challenging classification task was distinguishing between **jogging vs. running** due to kinematic similarity
- **Handclapping vs. handwaving** confusion was resolved well by sequential models (LSTM)

## ⚙️ Preprocessing Pipeline

1. **Sequence Length Analysis** — Visualized distribution of video frame counts (bimodal: ~100-150 and ~400-500 frames)
2. **Padding** — All sequences padded to match the longest video (824 frames)
3. **Scaling** — MinMax normalization applied to all features
4. **Train/Test Split** — 80/20 stratified split (479 training, 120 testing samples)
5. **Resampling** — For GAK+SVM, sequences resampled to 100 frames for computational feasibility

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) (only for feature extraction)
- CUDA-compatible GPU (recommended for training)

### Installation

```bash
pip install -r requirements.txt
```

### Dataset

The extracted feature files (`.npz`) can be downloaded from:
🔗 [Google Drive - Extracted Features](https://drive.google.com/file/d/1CJSfAfKf-7oFNLP99_0JM3MaIyXUdUBv/view?usp=sharing)

Place the downloaded files in the `extracted_features/` directory.

### Running

**Feature Extraction** (requires OpenPose installed):
```bash
python FeatureExtraction.py
```

**Model Training & Evaluation**:
```bash
python project.py
```

Or open the Jupyter notebooks for interactive execution with visualizations:
```bash
jupyter notebook project.ipynb
```

## 📚 Technologies Used

- **OpenPose** — Skeleton keypoint extraction (BODY_25 model)
- **tslearn** — Time series classification (GAK kernel, Shapelet Transform, resampling)
- **scikit-learn** — SVM classifier, evaluation metrics, data splitting
- **PyTorch** — Deep learning models (MLP, LSTM, 1D-CNN)
- **NumPy / Pandas / Matplotlib / Seaborn** — Data processing & visualization

## 📝 License

This project was developed as part of **AIN 313 - Introduction to Machine Learning** course assignment at Akdeniz University.
