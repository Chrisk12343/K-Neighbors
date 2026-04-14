# Titanic Survival Predictor

A GUI application that predicts Titanic passenger survival using machine learning models.

## Description

This application uses the Titanic dataset to train Decision Tree and K-Nearest Neighbors (KNN) models for predicting passenger survival. It provides a user-friendly Tkinter interface where users can input passenger details and receive predictions from both models.

## Features

- **Dual Model Predictions**: Get predictions from both Decision Tree and KNN classifiers
- **Interactive GUI**: Easy-to-use interface built with Tkinter
- **Model Accuracy Display**: Shows accuracy scores for both trained models
- **Feature Importance**: Displays the most important factors for survival prediction
- **Input Validation**: Handles invalid inputs gracefully

## Requirements

- Python 3.x
- pandas
- scikit-learn

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the `Titanic-Dataset.csv` file is in the same directory as `main.py`

## Usage

Run the application:
```bash
python main.py
```

The GUI will open with input fields for:
- Passenger Class (1-3)
- Sex (male/female)
- Age
- Number of Siblings/Spouses aboard

Click "Predict Survival" to see the predictions from both models.

## Models

- **Decision Tree Classifier**: Max depth of 4, trained on passenger class, sex, age, and siblings/spouses
- **K-Nearest Neighbors**: 5 neighbors, using the same features

## Dataset

The application uses the Titanic dataset (`Titanic-Dataset.csv`) which includes passenger information from the Titanic disaster. The dataset is preprocessed to handle missing values and convert categorical variables.

## How It Works

1. The dataset is loaded and preprocessed (filling missing ages, encoding sex)
2. Features are selected: Passenger Class, Sex, Age, Siblings/Spouses
3. Models are trained on 80% of the data and tested on 20%
4. User inputs are formatted and fed to both models for prediction
5. Results are displayed in the GUI

## Model Performance

The application displays the accuracy of both models on the test set. Typical accuracies are around 80-85% for both models with the selected features.
