# Discovering Emergent Visual Identities in Static Images

**Live Deployed App:** You can test the interactive web application live on the cloud here:  
[Launch Deployed Web App](https://discovering-emergent-visual-identities2026.streamlit.app/)

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

## Live Demo & Testing Guidance

The deployed web application provides an interactive interface to evaluate the pipeline's robustness on real-world data:

* **Random Image Inference:** Instantly analyze and view predictions for randomly sampled images from the validation pool.
* **External Image Upload:** Upload custom interior design images (`JPG`, `PNG`, `JPEG`, etc.) to predict their visual identity cluster in real-time.
* **Demo Evaluation (Unseen Data):** During evaluation and live demos, users can upload completely unseen interior design data to dynamically test the system's pattern recognition capability.

### Running the App Locally

If you wish to host and run the prototype on your local machine, follow these steps:

1. Clone the repository and open the project directory in your terminal.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
