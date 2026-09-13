   # Real vs. Fake Face Detection using Classical ML

   ## Project Overview
   This project classifies face images as real or AI-generated (StyleGAN) using handcrafted forensic features and classical machine learning. It proves that generation artifacts are detectable without using Deep Learning (CNNs).

   ## Dataset
   Uses a subset of the Kaggle "140k Real and Fake Faces" dataset (2,000 real Flickr images and 2,000 StyleGAN fake images).

   ## Methodology
   1. **Compression Bias Check:** Detects and neutralizes differences in JPEG compression between real and fake datasets.
   2. **Feature Extraction:** 
      - *Baseline:* Grayscale color histograms and basic statistics.
      - *Enhanced:* Frequency domain (DCT) and spatial noise residuals.
   3. **Classification:** Compares Support Vector Machines (SVM) and Random Forest classifiers.

   ## How to Run
   1. Install dependencies: `pip install opencv-python numpy scikit-learn scikit-image matplotlib seaborn`
   2. Place 2,000 real images in `real_subset/` and 2,000 fake images in `fake_subset/`.
   3. Run the script: `python3 main.py`
   4. View the generated `confusion_matrix.png` and `misclassified_samples.png`.

   ## Results
   The Random Forest classifier using Baseline features achieved the best performance (F1-Score: 0.6740). The project successfully identifies compression shortcuts and highlights the limitations of simple color features against highly photorealistic GANs.
