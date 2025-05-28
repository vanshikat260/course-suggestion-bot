import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("recommender_model.pkl")
le = joblib.load("label_encoder.pkl")

# Course catalog
course_catalog = [
    {"title": "React Basics", "category": "Frontend", "xp": 150},
    {"title": "Advanced CSS", "category": "Frontend", "xp": 120},
    {"title": "Node.js for Beginners", "category": "Backend", "xp": 180},
    {"title": "REST APIs with Express", "category": "Backend", "xp": 200},
    {"title": "Intro to Machine Learning", "category": "AI", "xp": 170},
    {"title": "Deep Learning with Python", "category": "AI", "xp": 250},
    {"title": "AI for Beginners", "category": "AI", "xp": 160},
    {"title": "Build a Portfolio Website", "category": "Frontend", "xp": 130},
    {"title": "Database Design Fundamentals", "category": "Backend", "xp": 190},
    {"title": "Computer Vision Projects", "category": "AI", "xp": 240},
]

# Prediction function
def predict_interest(user_xp, course_xp, course_cat, user_pref_cat):
    try:
        course_cat_encoded = le.transform([course_cat])[0]
        user_pref_cat_encoded = le.transform([user_pref_cat])[0]

        features = pd.DataFrame([{
            "user_xp": user_xp,
            "course_xp": course_xp,
            "course_cat_enc": course_cat_encoded,
            "user_pref_cat_enc": user_pref_cat_encoded
        }])

        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0

# Recommendation logic
def recommend_courses(completed_titles, user_pref_cat):
    total_xp = sum(course["xp"] for course in course_catalog if course["title"] in completed_titles)
    recommended = []

    for course in course_catalog:
        if course["title"] not in completed_titles:
            interested = predict_interest(total_xp, course["xp"], course["category"], user_pref_cat)
            if interested == 1:
                recommended.append(course["title"])
    return recommended

# --- Streamlit UI ---
st.title(" AI Course Recommender Chatbot")
st.markdown("Welcome to the smart course recommendation system based on your interests and learning progress.")

name = st.text_input(" Your name:")
interest = st.selectbox("Choose your area of interest:", ["AI", "Frontend", "Backend"])

st.markdown("### Select the courses you've already completed:")
completed_titles = st.multiselect(
    "Pick from the list:",
    [course["title"] for course in course_catalog]
)

if st.button("Get Recommendations"):
    if not name or not interest:
        st.warning("Please enter your name and select your interest.")
    else:
        st.success(f"Hi {name}! Here are some course recommendations for you in **{interest}**:")
        recommended = recommend_courses(completed_titles, interest)

        if recommended:
            for rec in recommended:
                st.write("~", rec)
        else:
            st.info("No strong recommendations found. Try selecting more completed courses or changing your interest.")

