# Voice Classification Advanced Model

[![GitHub Issues](https://img.shields.io/github/issues/dattanirjhar/multiclass-voice-recog)](https://github.com/dattanirjhar/multiclass-voice-recog/issues)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/maintenance/yes/2025)](https://github.com/dattanirjhar/multiclass-voice-recog/graphs/commit-activity)

## Overview

This repository hosts a multiclass voice recognition project like the previous multiclass classification model, enabling the classification of audio input into several distinct categories. It leverages state-of-the-art machine learning techniques to accurately identify and categorize different voice commands or speech patterns. This project aims to provide a robust and efficient solution for applications such as voice-controlled systems, automated transcription services, and advanced audio analysis.

Here we have used the previous model and tested it on the same song sung by different singers.

## Features

- **Multiclass Classification**: Accurately distinguishes between multiple voice commands or categories.
- **Elaborate Feature Extraction**: A total of 14 features have been extracted (out of which 11 are most essential and hgih result achieving) which are then statistically analyzed using numpy mean and standard deviation for efficient analysis.
- **Feature Dimension**: The extracted features are combined into a single 1-D array for model fitting (training). The combined features has an aggregate dimension ranging from 71 - 195 (depending on the features used and discarded).
- **High Accuracy**: Utilizes advanced machine learning models for optimal recognition rates.
- **Scalability**: Designed to handle a large number of voice samples and categories.
- **Customizable**: Easily adaptable to specific use cases and datasets.
- **Real-time Processing**: Optimized for real-time voice recognition applications.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Model Description](#model-description)
- [Evaluation](#evaluation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

- Python 3.7+
- [pip](https://pypi.org/project/pip/)
- [Virtualenv](https://virtualenv.pypa.io/en/latest/) (recommended)

### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/dattanirjhar/multiclass-voice-recog-V2.git
    cd multiclass-voice-recog-V2
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    virtualenv venv
    source venv/bin/activate   # On Linux/macOS
    venv\Scripts\activate.bat  # On Windows
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Data Preparation:**

    - Ensure your dataset is structured according to the guidelines in the [Dataset](#dataset) section.
    - Place your audio files in the appropriate directories.

2.  **Model Training:**

    Update the directory path (for dataset) in the Notebook(s).
    After the files get loaded, the script will split and train the model with the new dataset.
    The rest cells will need to be run inorder to train the model.

3.  **Inference/Prediction:**

    Run the required cells in the notebook to predict values based on the test split.

## Dataset

The project requires a structured dataset with audio samples organized into categories. Each category should have its own directory containing the corresponding audio files.

dataset/\
├──training/\
│ ├── category1\
│ │ ├── sample1.wav\
│ │ └── ...\
├── testing/\
│ ├── category1\
│ │ ├── sample1.wav\
│ │ └── ...\

- Ensure that the audio files are in a compatible format (e.g., WAV, MP3).
- Here we are using seven singers, each of whose vocal samples are colected (an average of 65 to 70 samples per singer)

## Model Architecture

A detailed description of the model architecture, including layers, activation functions, and key parameters.

- **Feature Extraction:** We use a sophisticated mechanism for audio analysis, designed to distill a wide array of acoustic characteristics from an audio file into a single, robust feature vector using the librosa library. It comprehensively captures timbral qualities through Mel-Frequency Cepstral Coefficients (MFCCs) and their dynamic changes (delta and delta-delta), harmonic content via Chroma and Constant-Q Transform (CQT) features, and tonal relationships using Tonnetz. Furthermore, it extracts fundamental vocal and sonic attributes like pitch stability (F0 standard deviation on voiced frames), brightness (spectral centroid), noisiness (zero-crossing rate), loudness (RMS energy), and fullness (spectral bandwidth). The function demonstrates a nuanced approach to feature engineering for enhanced accuracy by deliberately customizing the feature set; features like Spectral Contrast, Flatness, and Rolloff have been commented out. This selective exclusion is a critical optimization strategy to prevent feature redundancy and reduce the vector's dimensionality, thereby creating a more focused and discriminative set of features that can lead to better performance and faster training times for a machine learning model. The final output is a carefully curated and sanitized numerical vector, poised for effective use in complex audio classification or analysis tasks.
- **Model Type:** In building this model, we follow a robust workflow to tune a Support Vector Machine (SVM) classifier for optimal performance. We begin by preprocessing the data with RobustScaler, a technique that minimizes the influence of outliers by scaling features based on their interquartile range.

At the core of this modeling process is GridSearchCV, which we use to automate the critical task of finding the best model settings. Here, various combinations from a predefined param_grid are exhaustively tested to tune key hyperparameters like the regularization term C, the decision boundary type kernel (e.g., linear or rbf), and the kernel coefficient gamma. To ensure the resulting model is genuinely effective and not just overfitted, each combination is rigorously assessed using 5-fold cross-validation. Finally, the single best model identified from this search is used to make predictions on the unseen test set, providing a definitive accuracy score that reflects the model's true predictive power. 

## Model Description

The model involves the following steps:

1.  **Data Loading & Preprocessing:** Loading the raw audio files and processing each one using the extract_features_optimized function to generate a numerical feature vector. This complete dataset of features and corresponding labels is then split into training and testing sets. Finally, the RobustScaler is fitted on the training features and used to scale both the training and testing data, making the model resilient to outliers. 
2.  **Model Definition:** Defining the core classifier as a Support Vector Machine (SVC) from the scikit-learn library. This SVC instance is then encapsulated within a GridSearchCV object, which is configured with a specific param_grid containing the hyperparameters (C, kernel, gamma, etc.) to be tested.
3.  **Model Training:** After definition, the GridSearchCV object is fitted using the scaled training data. This single .fit() command triggers the complete training and optimization process, where it exhaustively searches for the best hyperparameter combination using 5-fold cross-validation to ensure the selected model is both accurate and robust.
4.  **Prediction & Evaluation:** After the grid search identifies the best-performing model, it is used to predict the classes for the unseen, scaled test set. The model's final performance is then evaluated by comparing these predictions against the true labels of the test set to calculate a final accuracy score.

- Monitor the training progress using the provided scripts and adjust parameters as needed.

## Evaluation

The model's performance is evaluated using metrics such as:

- **Accuracy**- 0.9016

Evaluation scripts are provided to assess the model's effectiveness on a test dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was inspired by advancements in machine learning and voice recognition technologies.
- Special thanks to the open-source community for providing valuable resources and tools.
