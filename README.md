# 🌿 Medicinal Plant Classification

An AI and Machine Learning based web application that identifies medicinal plants from uploaded images.

## 📌 Project Description

The **Medicinal Plant Classification** project uses image processing and Machine Learning to classify medicinal plants from their images.

The system extracts **HOG (Histogram of Oriented Gradients)** features from the uploaded image and uses an **SVM (Support Vector Machine)** classifier to predict the plant.

The project provides information about the predicted plant, including its traditional uses, benefits, and precautions.

## 🚀 Live Demo

👉 [Open Live App](https://medicinal-plant-classification-nc.streamlit.app)

## 🌱 Medicinal Plants

The model currently classifies five medicinal plants:

- Aloe Vera
- Hibiscus
- Neem
- Tulsi
- Moringa

## ✨ Features

- 🌿 Medicinal plant image classification
- 🤖 HOG + SVM Machine Learning model
- 📷 JPG, JPEG and PNG image upload
- 📊 Model confidence score
- 💊 Uses of the predicted plant
- 🌱 Benefits of the predicted plant
- ⚠️ Precautions
- 📋 Prediction history
- 🗑️ Clear prediction history
- 💻 Streamlit web interface
- 📱 Simple and user-friendly interface

## 🧠 Machine Learning

### Feature Extraction

The project uses **HOG (Histogram of Oriented Gradients)** for extracting visual features from plant images.

### Classification Algorithm

The extracted HOG features are classified using an **SVM (Support Vector Machine)** with an RBF kernel.

### Image Size

```text
128 × 128 pixels