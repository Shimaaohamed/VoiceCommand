# 🎤 Speech Commands Recognition: Voice Command Recognition System Using Multi-Input CNN

> 🚀 A high-performance AI system for speech command recognition with **scientific feature engineering** and **multi-input architecture**

---

## 🔥 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-NeuralNetworks-red?style=for-the-badge&logo=keras)
![Librosa](https://img.shields.io/badge/Librosa-AudioProcessing-purple?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-blue?style=for-the-badge&logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-ML-yellow?style=for-the-badge&logo=scikitlearn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green?style=for-the-badge)

</div>

---

## 📌 Overview

**Speech Commands Recognition** is an advanced AI system that:

* 🎤 Recognizes **10 voice commands** (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`)
* 🧠 Uses a **Multi-Input CNN** to fuse two different types of features
* 📊 Extracts scientifically justified audio features (MFCC + Spectral Centroid + ZCR)
* 🖼️ Generates a **Visual Pipeline** showing step-by-step transformations
* 🎯 Achieves accuracy exceeding **96%**

> ❗ Traditional models use only a single input
> ✅ This system fuses **Spectrogram (image)** with **Feature Vector (statistical)**

---

## 🖼️ System Architecture

<p align="center">
  <img src="images/multi_input_cnn_architecture.png" width="800"/>
</p>

---

## 🚀 Features

### 🧬 Scientific Feature Engineering

* **3 scientifically justified features** (15 values total):
  * **MFCC (13 coefficients):** Models the human vocal tract
  * **Spectral Centroid (1 value):** Represents sound "brightness"
  * **Zero Crossing Rate (1 value):** Distinguishes voiced vs. unvoiced sounds

* **Rejected features with justification:**
  * Chroma (designed for music analysis)
  * Harmonics (focuses on pitch, not content)
  * Spectral Roll-off (doesn't capture phonemes)

---

### 🖼️ Visual Pipeline

* Generates a 2×3 grid plot showing step-by-step transformations:
  1. **Raw Audio** → Original audio signal
  2. **Mel Spectrogram** → Spectrogram image
  3. **MFCC** → Coefficients
  4. **Mean MFCC** → Mean values
  5. **Spectral Centroid** → Sound brightness
  6. **ZCR** → Zero crossing rate

---

### 🤖 Multi-Input CNN Architecture

* **Branch 1 (CNN):** 4 Conv2D blocks (64→128→256→512) for the 128×128 spectrogram
* **Branch 2 (Dense):** 3 Dense layers for the 15-value feature vector
* **Merge:** Concatenate + Classification Head (128→64→10 Softmax)

---

### ⚡ Data Augmentation

* Time Masking
* Frequency Masking
* Gaussian Noise
* **Increases data from ~5000 to 6948 samples**

---

### 📊 Analytics & Monitoring

* Training/Validation Accuracy & Loss
* Confusion Matrix
* Classification Report (Precision, Recall, F1)
* Sample Spectrograms Visualization

---

## 📊 Performance Results

<p align="center">
  <img src="images/confusion_matrix.png" width="400"/>
  <img src="images/training_history.png" width="400"/>
</p>

### 🎯 Final Metrics

* **Accuracy:** 96.36% ✅ (exceeded 95% target)
* **F1 Score:** >0.95 for all commands
* **Precision:** >0.91 for all commands
* **Recall:** >0.93 for all commands
* **Total Samples:** 6948
* **Training Samples:** 5905
* **Testing Samples:** 1043

---

## 📋 Classification Report

| Command | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| down    | 0.95      | 0.97   | 0.96     | 105     |
| go      | 0.96      | 0.94   | 0.95     | 104     |
| left    | 0.94      | 0.98   | 0.96     | 103     |
| no      | 0.95      | 0.96   | 0.96     | 105     |
| off     | 0.91      | 0.99   | 0.95     | 105     |
| on      | 0.97      | 0.95   | 0.96     | 101     |
| right   | 1.00      | 0.95   | 0.98     | 106     |
| stop    | 0.97      | 0.98   | 0.98     | 105     |
| up      | 0.98      | 0.93   | 0.96     | 105     |
| yes     | 1.00      | 0.97   | 0.99     | 104     |
| **avg** | **0.96**  | **0.96** | **0.96** | **1043** |

🏆 **The model performs excellently on all commands**

---

## 🤖 Detailed Model Architecture

<p align="center">
  <img src="images/cnn_blocks.png" width="600"/>
</p>

### Branch 1: CNN for Spectrogram

| Block | Layers | Filters | Output Shape |
|-------|--------|---------|--------------|
| Block 1 | Conv2D×2 + BN + MaxPool + Dropout | 64 | 128×128 → 64×64 |
| Block 2 | Conv2D×2 + BN + MaxPool + Dropout | 128 | 64×64 → 32×32 |
| Block 3 | Conv2D×2 + BN + MaxPool + Dropout | 256 | 32×32 → 16×16 |
| Block 4 | Conv2D×2 + BN + GAP + Dropout | 512 | 16×16 → 512 |
| Dense | Dense + BN + Dropout | 256 | 256 |

### Branch 2: Dense for Features

| Layer | Size | Activation |
|-------|------|------------|
| Dense 1 | 64 | ReLU |
| Dense 2 | 32 | ReLU |
| Dense 3 | 32 | ReLU |

### Classification Head

* Concatenate (256 + 32 = 288)
* Dense (128) → BN → Dropout
* Dense (64) → BN → Dropout
* Output (10) → Softmax

---

## 🗂️ Dataset

<p align="center">
  <img src="images/dataset_samples.png" width="600"/>
</p>

**Speech Commands Dataset v0.02** contains:

* 🎤 10 core voice commands
* 👥 Recordings from diverse speakers
* 📁 `.wav` files at 16kHz
* ⏱️ ~1 second per recording

---

## ⚙️ Installation

### Prerequisites

* Python 3.8+
* pip
* GPU (recommended for speed)

### Setup

```bash
git clone <your-repo-link>
cd Speech-Commands-Recognition
pip install -r requirements.txt
```

### Main Libraries

```bash
pip install librosa seaborn matplotlib scipy scikit-learn tensorflow
```

---

## ▶️ Run the Project

```bash
jupyter notebook final_speech_project.ipynb
```

Or directly in Colab:

```python
# Clone the project
!git clone <your-repo-link>
%cd Speech-Commands-Recognition

# Run the notebook
!jupyter nbconvert --to notebook --execute final_speech_project.ipynb
```

---

## 📁 Project Structure

```
Speech-Commands-Recognition/
│
├── speech_commands_v0.02/    # Dataset
├── models/                    # Trained models
├── outputs/                   # Results & plots
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── sample_spectrograms.png
│   ├── pipeline_yes.png
│   ├── pipeline_no.png
│   └── pipeline_up.png
│
├── final_speech_project.ipynb  # Main notebook
├── best_multi_cnn.keras       # Best saved model
├── requirements.txt
└── README.md
```

---

## 🎯 Use Cases

* 🏠 **Smart Home Devices:** Voice control
* 📱 **Interactive Apps:** Command recognition
* 🎮 **Voice-Controlled Games:** Interactive experience
* ♿ **Accessibility Tools:** Voice control for people with disabilities
* 📊 **Research Projects:** Audio signal processing studies

---

## 🏆 Highlights

* ✅ **End-to-end AI pipeline** from start to finish
* ✅ **Scientifically justified feature engineering** with academic references
* ✅ **Advanced Multi-Input CNN architecture**
* ✅ **Visual Pipeline** for clear illustration
* ✅ **96%+ accuracy** across 10 classes
* ✅ **Data Augmentation** for better generalization
* ✅ **Early Stopping & LR Scheduling** for training optimization

---

## 🔬 Comparison with Other Approaches

| Approach | Input | Accuracy |
|----------|-------|----------|
| Traditional ML (SVM) | MFCC only | ~85% |
| Single-Input CNN | Spectrogram only | ~92% |
| **Multi-Input CNN** | **Spectrogram + Features** | **96.36%** ✅ |

---

## 🎯 Conclusion

This project combines:

* 🧠 **AI Performance:** 96.36% accuracy across 10 classes
* 🎨 **Scientific Feature Engineering:** MFCC + Centroid + ZCR
* 🤖 **Advanced Architecture:** Multi-Input CNN
* 📊 **Comprehensive Documentation:** Visual Pipeline + Analytics

➡️ Making it a **professional, research-ready voice recognition system**

---

<div align="center">

### ⭐ Professional • High-Performance • Research-Ready

</div>