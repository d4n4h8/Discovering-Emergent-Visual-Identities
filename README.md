# 🏠 Discovering Emergent Visual Identities in Static Images

This repository contains an AI-based unsupervised computer vision system designed to automatically discover emergent visual identities in static interior design images without relying on predefined labels. 

---

### 🌐 Live Deployment
🚀 **Test the interactive web application live on the cloud here:**  
👉 [Launch Deployed Web App](https://discovering-emergent-visual-identities2026.streamlit.app/)

---

### 📌 Project Overview & Goal
Traditional computer vision heavily relies on human-labeled datasets, limiting a system's capacity to recognize unknown patterns. This project addresses this limitation by deploying an **Unsupervised Learning** pipeline to explore hidden visual patterns, textures, and layouts across **5,500 interior design images** (sampled randomly from the *Interior Design Styles Dataset* on Kaggle).

---

### ⚙️ System Pipeline Architecture
The system follows a modular pipeline consisting of 9 interconnected stages:
1. **Data Acquisition:** Fetching raw interior images via KaggleHub.
2. **Data Loading & Preprocessing:** Shuffling, validating, resizing to $224 \times 224$ pixels, and normalizing inputs to $[0, 1]$.
3. **Hybrid Feature Extraction:** 
   * **Deep Semantic Features:** Extracted via a pre-trained **ResNet50** CNN.
   * **Handcrafted Features:** RGB Color Histograms + Local Binary Patterns (LBP) for texture.
4. **Feature Standardization:** Equalizing distributions using `StandardScaler`.
5. **Dimensionality Reduction:** Applying **PCA** (retaining 200 Principal Components) to boost efficiency.
6. **Clustering Exploration:** Benchmarking K-Means, Agglomerative Clustering, and DBSCAN.
7. **Visualization & Interpretability:** Mapping feature spaces via **t-SNE**, Heatmaps, and Dendrograms.
8. **Evaluation:** Silhouette Score analysis.
9. **Inference Engine:** Real-time cluster assignment for unseen or uploaded images.

---

### 🛠️ Technologies & Libraries
<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow%20/%20Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy">
</p>

---

### 📊 Experimental Results & Model Evaluation

#### 1. Algorithmic Comparison
We tested multiple clustering approaches to find the optimal boundary definition:
* 🏆 **K-Means ($K=3$):** Selected as the baseline model. Provided the highest structural stability with a **Silhouette Score of 0.1222**.
* **Agglomerative Clustering:** Achieved a silhouette score of 0.1110 (less effective at dense boundary separation).
* **DBSCAN:** Achieved a score of 0.3140 by isolating 259 anomalous samples as noise, which improved dataset hygiene.

#### 2. Discovered Identities Distribution
The unsupervised model automatically categorized the data into 3 dominant visual clusters:
* **Cluster 0:** 2,868 images (Strongly correlated with *Transitional* styles)
* **Cluster 1:** 1,690 images (Strongly correlated with *Eclectic* styles)
* **Cluster 2:** 942 images (Strongly correlated with *Tropical* styles)

*Note: Heatmap analysis verified that even without training labels, the clusters tightly align with real-world human design concepts.*

---

### 🚀 Live Demo & Testing Guidance
The interactive **Streamlit App** allows you to test the model dynamically via two inference modes:
1. **Random Image Inference:** Samples an image from the unseen validation pool and predicts its visual identity instantly.
2. **External Image Upload:** Upload any custom interior design image (`.jpg`, `.png`) to evaluate the model's feature extraction and mapping robustness in real-time.

---

### 💻 Local Installation & Setup
To run the Streamlit dashboard locally on your machine, execute the following commands:

```bash
# 1. Clone the repository
git clone [https://github.com/d4n4h8/Discovering-Emergent-Visual-Identities.git](https://github.com/d4n4h8/Discovering-Emergent-Visual-Identities.git)

# 2. Navigate to the project folder
cd Discovering-Emergent-Visual-Identities

# 3. Install necessary dependencies
pip install -r requirements.txt

# 4. Run the Streamlit web server
streamlit run app.py
