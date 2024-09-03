# Fire Detection System

Welcome to the Fire Detection System repository. This project aims to detect fire in images using machine learning models. <br>
The repository includes a Jupyter notebook that describes the data, builds and trains three models, evaluates their performance,
performs error analysis, and explains the results using explainable AI (XAI) techniques. 
Additionally, a demo video showcases the potential use of the fire detection model.

## Repository Contents

- `Fire_Detection.ipynb`: A comprehensive Jupyter notebook that covers the entire process from data exploration to model building and evaluation.
- `video.mp4`: A demonstration video showing the potential use of the fire detection system.

## Notebook Overview

### 1. Data Description
- Overview of the dataset used for training and testing the fire detection models.
- Visualizations and statistics to understand the distribution and characteristics of the data.

### 2. Building and Training Models
- Implementation and training of three different machine learning models for fire detection:
  1. Model 1: Convolutional Neural Network (CNN)
  2. Model 2: Support Vector Machine (SVM)
  3. Model 3: Random Forest Classifier
- Description of the architecture and hyperparameters used for each model.

### 3. Metrics
- Evaluation metrics used to assess the performance of the models, including accuracy, precision, recall, and F1-score.
- Comparison of model performance based on these metrics.

### 4. Error Analysis
- Analysis of the misclassified instances to understand the failure modes of the models.
- Insights into the types of images or features that are challenging for the models to correctly classify.

### 5. Explainable AI (XAI)
- Techniques used to explain the model predictions.
- Visualization of feature importance and model decision-making process to ensure transparency and trust in the model.

## Demo Video

The `video.mp4` file contains a demonstration of the fire detection system in action. It illustrates the practical application of the model in a real-world scenario, highlighting its potential to enhance fire safety and prevention measures.

## Requirements

Ensure you have the following dependencies installed:
- Python 3.x
- Jupyter Notebook
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow/keras, and any other dependencies mentioned in the notebook

## Acknowledgements

Special thanks to the creators of the dataset and the open-source community for providing the tools and resources necessary for this project.
