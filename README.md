# AI Course Recommender Bot

An intelligent chatbot that recommends tech courses (AI, Frontend, Backend) to users based on their preferences and previously completed courses. The model is trained on dummy user-course interaction data using a `RandomForestClassifier` and presented in a user-friendly interface built with Streamlit.

## 🚀 Features

- Understands user’s preferred category (AI, Frontend, Backend)
- Recommends relevant courses based on experience and completed titles
- Machine learning-powered predictions (Random Forest)
- Chat-like interface using Streamlit
- Modular structure: model and chatbot logic separated into different files


---

## How It Works

1. `course_recommender.py`:
   - Trains a recommendation model using synthetic data
   - Saves the trained model and label encoder

2. `app.py`:
   - Runs a chatbot interface where users:
     - Enter their name and area of interest
     - Select completed courses from a predefined catalog
     - Receive tailored course suggestions



