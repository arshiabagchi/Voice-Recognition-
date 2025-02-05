
# Voice Recognition System

## Overview

The **Voice Recognition System** is a machine learning project designed to identify speakers based on their unique voice patterns. This system leverages **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction and a **neural network classifier** built using **TensorFlow/Keras** for speaker identification. 

This project showcases the power of machine learning in the field of audio and speech recognition and can be used in applications such as **voice-based authentication**, **security**, and **personal assistants**.

## Key Features

- **Voice Sample Capture**: The system allows users to record voice samples directly from the browser.
- **MFCC Feature Extraction**: The captured voice samples are processed to extract **MFCC features**, which represent the essential characteristics of speech.
- **Neural Network Model**: A deep learning model built using **TensorFlow/Keras** is used to classify different speakers.
- **Real-Time Prediction**: The system can predict the speaker's identity based on the recorded voice sample.
- **Easy-to-Use Interface**: An intuitive interface for collecting and processing voice data.

## Technologies Used

- **Python**: The core programming language for data processing, training, and evaluation.
- **TensorFlow/Keras**: For building and training the machine learning model.
- **Librosa**: For audio processing and feature extraction.
- **Google Colab**: For running the code and training the model in the cloud.
- **Matplotlib**: For visualizing the results.
- **NumPy**: For handling arrays and datasets.
- **JavaScript**: For recording voice samples directly from the user's browser.

## How It Works

### 1. **Audio Capture**

Voice samples are recorded directly from the user's microphone. Once recorded, the audio is saved as **.wav** files, which are stored in **Google Drive**.

### 2. **MFCC Feature Extraction**

Once the audio samples are captured, **MFCC features** are extracted using the **Librosa** library. These features represent key characteristics of the voice and are used as input to the neural network model.

```python
import librosa
import numpy as np

# Function to extract MFCC features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc, axis=1)  # Average across the time axis
    return mfcc
