# Multiclass Voice Recognition V2

[![GitHub Issues](https://img.shields.io/github/issues/dattanirjhar/multiclass-voice-recog)](https://github.com/dattanirjhar/multiclass-voice-recog/issues)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/maintenance/yes/2025)](https://github.com/dattanirjhar/multiclass-voice-recog/graphs/commit-activity)

## Overview

This repository hosts a multiclass voice recognition project like the previous multiclass classification model, enabling the classification of audio input into several distinct categories. It leverages state-of-the-art machine learning techniques to accurately identify and categorize different voice commands or speech patterns. This project aims to provide a robust and efficient solution for applications such as voice-controlled systems, automated transcription services, and advanced audio analysis.

Here we have used the previous model and tested it on the same song sung by different singers.

## Features

- **Multiclass Classification**: Accurately distinguishes between multiple voice commands or categories.
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
- [Training](#training)
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

- **Feature Extraction:** Describe the audio feature extraction techniques used (e.g., MFCC, Spectrogram).
- **Model Type:** We use an SVM model here, with a Linear Kernel, and OVR decision function.

## Training

The training process involves the following steps:

1.  **Data Loading:** Loading the dataset and preprocessing the audio samples and splitting into training and testing sets.
2.  **Model Definition:** Defining and calling the SVM model from **scikit-learn** library.
3.  **Model Training:** After definition, we fit the training set on the model and train it.
4.  **Prediction:** After training, we predict values(classes) on the test set.

- Monitor the training progress using the provided scripts and adjust parameters as needed.

## Evaluation

The model's performance is evaluated using metrics such as:

- **Accuracy**- 0.82

Evaluation scripts are provided to assess the model's effectiveness on a test dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was inspired by advancements in machine learning and voice recognition technologies.
- Special thanks to the open-source community for providing valuable resources and tools.
