# Voice-Recognition

## Overview  
This project is a **voice recognition system** that allows users to create a dataset of labeled voice samples, train a deep learning model, and recognize speakers in real-time using voice input. The dataset is stored in **Google Drive**, ensuring persistence across different sessions.

## Features  
- Capture and store labeled voice samples in Google Drive  
- Train a neural network model for voice recognition using MFCC features  
- Use TensorFlow and Keras for model training  
- Visualize and preprocess voice data  
- Predict and recognize speakers in real-time  

## Technologies Used  
- Python (Google Colab)  
- TensorFlow/Keras  
- Librosa (for audio feature extraction)  
- Matplotlib  
- Google Drive API  

## Project Workflow  
1. **Dataset Creation**  
   - Mounts Google Drive and stores voice samples in a specified directory.  
   - Uses the microphone to capture labeled voice samples and save them to the dataset.  
   - Ensures each person has at least one voice sample.  

2. **Data Preprocessing**  
   - Loads audio samples and extracts **MFCC** features using **Librosa**.  
   - Normalizes the MFCC features for better model performance.  
   - Splits data into training and testing sets using `train_test_split`.  

3. **Model Training**  
   - Uses a **neural network** model with dense layers for classification.  
   - Trains the model using **sparse categorical cross-entropy** loss and **Adam** optimizer.  
   - Evaluates the trained model on the test dataset.  

4. **Voice Recognition**  
   - Records a new voice sample.  
   - Preprocesses the sample and feeds it into the trained model.  
   - Predicts and displays the recognized speaker.  

## How to Use  
1. **Run the Notebook in Google Colab**  
   - Ensure that **Google Drive** is properly mounted.  
   - Run the dataset creation script to capture voice samples.  

2. **Train the Model**  
   - Execute the training script and monitor the model's accuracy.  
   - Adjust hyperparameters and retrain if necessary.  

3. **Recognize Speakers**  
   - Capture a new voice sample and run it through the trained model.  
   - The model will predict and display the recognized speaker.  

## Acknowledgments  
This project is inspired by voice recognition applications and aims to provide an effective implementation using deep learning techniques.
