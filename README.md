# Discovering Emergent Visual Identities in Static Images

## Project Overview
This project is an AI-based unsupervised computer vision system designed to discover emergent visual identities in static interior design images.

The system analyzes image datasets, extracts meaningful visual features, groups similar images into clusters, and visualizes the discovered identities.

## Project Goal
The goal of this project is to identify hidden visual patterns and design identities without relying on predefined labels.

Instead of traditional image classification, the system uses unsupervised learning to discover visual similarities automatically.

## Dataset
The project uses the Interior Design Styles Dataset from Kaggle.

The dataset contains interior design images from multiple design styles and provides a diverse collection of colors, layouts, textures, and visual patterns.

## System Pipeline
The system pipeline consists of:

1. Data Acquisition
2. Data Exploration
3. Image Preprocessing
4. Feature Extraction
5. Dimensionality Reduction
6. Clustering
7. Visualization
8. Evaluation
9. Demo / Inference

## Technologies and Libraries
The project uses:

- Python
- KaggleHub
- OpenCV
- NumPy
- Matplotlib
- Scikit-image
- TensorFlow / Keras
- Scikit-learn
- Seaborn

## Feature Extraction
The system extracts hybrid visual features using:

- ResNet50 for deep visual features
- Color histograms
- Local Binary Patterns (LBP)

## Baseline Model
For the baseline implementation, the system uses:

- PCA for dimensionality reduction
- K-Means as the baseline clustering model

## Evaluation
The system is evaluated using:

- Silhouette Score
- t-SNE visualization
- Representative cluster images
- Heatmap analysis

## Demo
The project includes:
- Random image inference
- External image upload for prediction
  
Optional: upload an interior design image during the demo section to test the system on unseen data.
2. Install required libraries:

```bash
pip install -r requirements.txt
