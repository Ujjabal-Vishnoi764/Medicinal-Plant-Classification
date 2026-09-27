# 🌿 Medicinal Plant Classification

An AI and Machine Learning based web application that identifies medicinal plants from uploaded images.

## 📌 Project Description

The **Medicinal Plant Classification** project uses image processing and Machine Learning to classify medicinal plants from their images.

The system extracts **HOG (Histogram of Oriented Gradients)** features from the uploaded image and uses an **SVM (Support Vector Machine)** classifier to predict the plant.

The project provides information about the predicted plant, including its traditional uses, benefits, and precautions.

## 🌱 Medicinal Plants

The model currently classifies five medicinal plants:

* Aloe Vera
* Hibiscus
* Neem
* Tulsi
* Moringa

## ✨ Features

* 🌿 Medicinal plant image classification
* 🤖 HOG + SVM Machine Learning model
* 📷 JPG, JPEG and PNG image upload
* 📊 Model confidence score
* 💊 Uses of the predicted plant
* 🌱 Benefits of the predicted plant
* ⚠️ Precautions
* 📋 Prediction history
* 🗑️ Clear prediction history
* 💻 Streamlit web interface
* 📱 Simple and user-friendly interface

## 🧠 Machine Learning

### Feature Extraction

The project uses **HOG (Histogram of Oriented Gradients)** for extracting visual features from plant images.

### Classification Algorithm

The extracted HOG features are classified using an **SVM (Support Vector Machine)** with an RBF kernel.

### Image Size

```text
128 × 128 pixels
```

### Model Accuracy

```text
75.86%
```

The accuracy is based on the project's test split.

> Note: The displayed confidence is a model score converted into a percentage-like value. It is not a calibrated probability.

## 🛠️ Technologies Used

* Python
* Streamlit
* NumPy
* Pillow
* Scikit-image
* Scikit-learn
* Joblib
* HOG
* SVM
* VS Code
* Git & GitHub

## 📂 Project Structure

```text
Medicinal_Plant_Classification/
│
├── model/
│   ├── plant_classifier.pkl
│   └── label_encoder.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── dataset/
│
├── app.py
├── streamlit_app.py
├── train_model.py
├── plant_info.py
├── requirements.txt
├── .gitignore
└── README.md
```

> The trained model files and dataset are kept locally and are not included in the GitHub repository because of their large size.

## ▶️ How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/Ujjabal-Vishnoi764/Medicinal-Plant-Classification.git
```

### 2. Open the project

```bash
cd Medicinal-Plant-Classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

For Windows PowerShell:

```bash
venv\Scripts\Activate.ps1
```

### 5. Install required packages

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run streamlit_app.py
```

The application will open in the browser.

## 🔄 Model Training

The model can be trained using:

```bash
python train_model.py
```

The training process:

```text
Plant Image
     ↓
Resize to 128 × 128
     ↓
Grayscale Conversion
     ↓
HOG Feature Extraction
     ↓
SVM Classification
     ↓
Predicted Plant
```

## 📊 Application Workflow

```text
Upload Plant Image
        ↓
Image Preprocessing
        ↓
HOG Feature Extraction
        ↓
SVM Model
        ↓
Plant Prediction
        ↓
Plant Information
        ↓
Prediction History
```

## ⚠️ Disclaimer

The plant information provided by this project is for **educational purposes only**.

The information should not be considered medical advice. Always consult a qualified healthcare professional before using any plant or herbal preparation for medicinal purposes.

## 👨‍💻 Developer

**Ujjabal Vishnoi**

MCA Student
Teerthanker Mahaveer University (TMU)

## ⭐ Project

This project was developed as an MCA Machine Learning mini-project.
