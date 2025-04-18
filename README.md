# 🧠 Sarcasm Detector (Django + ML)

A web application that detects sarcasm in a sentence using a trained machine learning model. Built with Django and scikit-learn.

## 🚀 Features

- Predicts whether a sentence is sarcastic or not
- Clean user interface built with HTML/CSS
- Machine learning model trained with TF-IDF and classification algorithm
- Preprocessing of input for better prediction accuracy

## 🖥️ Screenshots

![App Screenshot](screenshot.png) *(Add your screenshot here)*

## ⚙️ Tech Stack

- Python
- Django
- scikit-learn
- HTML/CSS
- Joblib (for model loading)

## 📦 Project Structure

sarcasm-detector/ │ ├── detector/ # Django app │ ├── views.py # Prediction logic │ ├── templates/ │ │ └── index.html # Frontend │ └── text_preprocessing.py# Cleaning function │ ├── models/ # Trained model and vectorizer │ ├── sarcasm_model.pkl │ └── tfidf_vectorizer.pkl │ ├── sarcasm_project/ # Django project files │ └── settings.py │ ├── db.sqlite3 # Database └── manage.py