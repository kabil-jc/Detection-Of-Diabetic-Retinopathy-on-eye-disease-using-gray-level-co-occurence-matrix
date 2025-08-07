# 🧠 Detection of Diabetic Retinopathy using Gray Level Co-occurrence Matrix (GLCM)

## 📌 Overview
This project aims to detect **Diabetic Retinopathy (DR)** using **machine learning** and **texture-based image analysis**. By leveraging the **Gray Level Co-occurrence Matrix (GLCM)**, the model extracts key texture features from retinal fundus images and classifies them into different stages of diabetic retinopathy.

---

## 🔍 Problem Statement
Diabetic Retinopathy is a diabetes complication that affects eyes and can cause blindness if not detected early. Manual detection is time-consuming and requires expert knowledge. This project automates the detection process using **image processing** and **ML classifiers**, helping doctors diagnose DR efficiently.

---

## 🎯 Objectives
- Preprocess retinal images (noise removal, grayscale conversion)
- Extract texture features using **GLCM**
- Train ML classifiers: **Decision Tree**, **Random Forest**, and **XGBoost**
- Classify DR into **five severity levels**
- Achieve high accuracy in classification

---

## 🛠️ Tech Stack

- **Programming Language**: Python  
- **Libraries**:  
  - `OpenCV` – image preprocessing  
  - `scikit-image` – GLCM feature extraction  
  - `scikit-learn` – ML models & evaluation  
  - `XGBoost` – advanced gradient boosting  
- **Platform**: Jupyter Notebook

---

## ⚙️ Image Processing Workflow

1. **Load Image**  
2. **Convert to Grayscale**  
3. **Denoise / Resize**  
4. **Apply GLCM (Gray Level Co-occurrence Matrix)**  
5. **Extract Features**  
6. **Feed into Classifiers**  
7. **Predict Stage of Diabetic Retinopathy**

---

## 📊 Features Extracted (GLCM)
- **Contrast** – Measures local intensity variation
- **Dissimilarity** – Difference between pixel pairs
- **Homogeneity** – Similarity of elements
- **Energy (ASM)** – Uniformity
- **Correlation** – Relationship between pixel pairs

---

## 🧠 Machine Learning Models

| Model           | Accuracy (%) |
|----------------|---------------|
| Decision Tree  | ~80%          |
| Random Forest  | ~85%          |
| XGBoost        | **87%**       |

✅ *XGBoost outperformed other models in accuracy and consistency.*

---

## 📂 Folder Structure
```bash
diabetic-retinopathy-glcm/
├── data/
│   └── fundus_images/
├── notebooks/
│   └── GLCM_Feature_Extraction.ipynb
├── models/
│   └── trained_models.pkl
├── utils/
│   └── preprocessing.py
├── README.md
